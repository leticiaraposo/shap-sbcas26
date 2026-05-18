# =============================================================================
# SHAP - Estratificação de Risco Cardiovascular
# Capítulo: Interpretação de Modelos de ML Aplicados à Saúde com SHAP
# Linguagem: R
# Ambiente: RStudio
# =============================================================================

# -----------------------------------------------------------------------------
# Instalação de dependências
# -----------------------------------------------------------------------------
# install.packages(c("data.table", "xgboost", "shapviz", "pROC", "ggplot2"),
#                  repos = "https://cloud.r-project.org", quiet = TRUE)


# -----------------------------------------------------------------------------
# Carregamento das bibliotecas
# -----------------------------------------------------------------------------
library(data.table)
library(xgboost)
library(shapviz)
library(pROC)
library(ggplot2)

cat("Bibliotecas carregadas com sucesso.\n")
cat(sprintf("  xgboost  %s\n", packageVersion("xgboost")))
cat(sprintf("  shapviz  %s\n", packageVersion("shapviz")))

# -----------------------------------------------------------------------------
# Carregamento e limpeza dos dados
# -----------------------------------------------------------------------------
df <- fread("cardio_train.csv", sep = ";")   # separador padrão do dataset

# Limpeza: pressão arterial
df <- df[
  ap_hi >= 70 & ap_hi <= 250 &
  ap_lo >= 40 & ap_lo <= 150 &
  ap_hi > ap_lo
]

# Altura e peso
df <- df[
  height >= 120 & height <= 220 &
  weight >= 40  & weight <= 200
]

# Engenharia de atributos
df[, age_years := age / 365.25]
df[, imc       := weight / (height / 100)^2]
df <- df[imc >= 15 & imc <= 60]

cat(sprintf("N após limpeza: %s observações\n", format(nrow(df), big.mark = ".")))

# -----------------------------------------------------------------------------
# Definição de variáveis e particionamento
# -----------------------------------------------------------------------------
features <- c(
  "age_years", "gender", "imc",
  "ap_hi", "ap_lo",
  "cholesterol", "gluc",
  "smoke", "alco", "active"
)
target <- "cardio"

stopifnot(all(c(features, target) %in% names(df)))

X <- as.matrix(df[, ..features])
y <- as.numeric(df[[target]])

# Particionamento estratificado (75% treino / 25% teste)
set.seed(42)
idx_pos  <- which(y == 1)
idx_neg  <- which(y == 0)
idx_test <- c(
  sample(idx_pos, size = floor(0.25 * length(idx_pos))),
  sample(idx_neg, size = floor(0.25 * length(idx_neg)))
)

X_train <- X[-idx_test, , drop = FALSE];  y_train <- y[-idx_test]
X_test  <- X[ idx_test, , drop = FALSE];  y_test  <- y[ idx_test]

dtrain <- xgb.DMatrix(data = X_train, label = y_train)
dtest  <- xgb.DMatrix(data = X_test,  label = y_test)

cat(sprintf("Treino: %s | Teste: %s\n",
            format(length(y_train), big.mark = "."),
            format(length(y_test),  big.mark = ".")))
cat(sprintf("Prevalência (teste): %.3f\n", mean(y_test)))

# -----------------------------------------------------------------------------
# Treinamento do modelo XGBoost
# -----------------------------------------------------------------------------
params <- list(
  objective        = "binary:logistic",
  eval_metric      = "logloss",
  max_depth        = 4,
  eta              = 0.05,
  subsample        = 0.8,
  colsample_bytree = 0.8,
  lambda           = 1.0,
  alpha            = 0.1
)

model <- xgb.train(
  params    = params,
  data      = dtrain,
  nrounds   = 300,
  watchlist = list(test = dtest),
  verbose   = 0
)

cat("Treinamento concluído.\n")

# -----------------------------------------------------------------------------
# Avaliação preditiva
# -----------------------------------------------------------------------------
pred_prob <- predict(model, dtest)
auc_val   <- as.numeric(pROC::auc(y_test, pred_prob, quiet = TRUE))

# Brier Score
brier <- mean((pred_prob - y_test)^2)

cat(sprintf("AUC-ROC    : %.4f\n", auc_val))
cat(sprintf("Brier Score: %.4f\n", brier))

# -----------------------------------------------------------------------------
# Cálculo dos valores SHAP (TreeSHAP via predcontrib)
# -----------------------------------------------------------------------------
# Método 1: via predict com predcontrib (log-odds; última coluna = BIAS)
shap_contrib <- predict(model, dtest, predcontrib = TRUE)
shap_matrix  <- shap_contrib[, 1:(ncol(shap_contrib) - 1), drop = FALSE]
base_value   <- shap_contrib[1, ncol(shap_contrib)]
colnames(shap_matrix) <- features

cat(sprintf("Valor base φ₀: %.4f  (escala log-odds)\n", base_value))

# Verificação da decomposição aditiva para a primeira observação
pred_logodds    <- base_value + sum(shap_matrix[1, ])
pred_prob_check <- 1 / (1 + exp(-pred_logodds))
cat(sprintf("\nVerificação (observação 1):\n"))
cat(sprintf("  Predição reconstruída via SHAP : %.4f\n", pred_prob_check))
cat(sprintf("  Predição direta do modelo      : %.4f\n", pred_prob[1]))

# Método 2: objeto shapviz (usado nas visualizações abaixo)
sv <- shapviz(model, X_pred = dtest, X = X_test)

# -----------------------------------------------------------------------------
# Bar plot: ranking de importância global
# -----------------------------------------------------------------------------
p_bar <- sv_importance(sv, kind = "bar") +
  labs(title = "Importância Global — SHAP",
       x     = expression(paste("Média de ", "|", phi[i], "|")),
       y     = "Variável") +
  theme_minimal(base_size = 14)

ggsave("fig_importancia_shap.png",
       plot   = p_bar,
       width  = 16, height = 12, units = "cm",
       dpi    = 600)

# -----------------------------------------------------------------------------
# Summary plot (beeswarm): importância + direção dos efeitos
# -----------------------------------------------------------------------------
p_beeswarm <- sv_importance(sv, kind = "beeswarm") +
  labs(title = "Summary Plot (Beeswarm) — SHAP") +
  theme_minimal(base_size = 14)

ggsave("fig_beeswarm_shap.png", p_beeswarm,
       width = 16, height = 16, units = "cm",
       dpi = 600)

# -----------------------------------------------------------------------------
# Dependence plot: ap_hi × colesterol
# -----------------------------------------------------------------------------
p_dep <- sv_dependence(sv, v = "ap_hi", color_var = "cholesterol") +
  labs(
    title = "Dependence Plot — Pressão Arterial Sistólica",
    x     = "ap_hi (mmHg)",
    y     = expression(paste("Valor SHAP de ap_hi (", phi[i], ")"))
  ) +
  theme_minimal(base_size = 14)

ggsave("fig_dependence_aphi.png", p_dep,
       width = 16, height = 12, units = "cm", 
       dpi= 600)

# -----------------------------------------------------------------------------
# Waterfall plot: explicação local para um paciente específico
# -----------------------------------------------------------------------------
idx_paciente <- 1   # altere para inspecionar outro paciente (base 1 em R)

p_waterfall <- sv_waterfall(sv, row_id = idx_paciente) +
  labs(title = sprintf("Waterfall Plot — Paciente %d", idx_paciente)) +
  theme_minimal(base_size = 14)

ggsave("fig_waterfall_paciente1.png", p_waterfall,
       width = 14, height = 10, units = "cm",
       dpi = 600)

# Tabela de contribuições
contrib_df <- data.frame(
  Variavel            = features,
  Valor_observado     = X_test[idx_paciente, ],
  Contribuicao_SHAP   = shap_matrix[idx_paciente, ]
)
contrib_df <- contrib_df[order(abs(contrib_df$Contribuicao_SHAP), decreasing = TRUE), ]
rownames(contrib_df) <- NULL

cat(sprintf("\nDecomposição SHAP — Paciente %d:\n", idx_paciente))
cat(sprintf("  Valor base φ₀ : %.4f\n", base_value))
cat(sprintf("  Predição final: %.4f (log-odds) → %.4f (probabilidade)\n",
            pred_logodds, pred_prob_check))
cat("\n")
print(contrib_df, digits = 4)

# -----------------------------------------------------------------------------
# Force plot: visualização compacta da explicação local
# -----------------------------------------------------------------------------
tiff("fig_force_paciente1.png",
     width = 7, height = 5, units = "in", res = 600)

sv_force(sv, row_id = idx_paciente) +
  labs(title = sprintf("Force Plot — Paciente %d", idx_paciente))

dev.off()

# -----------------------------------------------------------------------------
# Importância global: tabela ordenada por |SHAP médio|
# -----------------------------------------------------------------------------
mean_abs_shap <- colMeans(abs(shap_matrix))
imp_df <- data.frame(
  Variavel                    = features,
  Importancia_global          = mean_abs_shap
)
imp_df <- imp_df[order(imp_df$Importancia_global, decreasing = TRUE), ]
rownames(imp_df) <- NULL

cat("Ranking de importância global (E[|φᵢ|]):\n")
print(imp_df, digits = 4)
