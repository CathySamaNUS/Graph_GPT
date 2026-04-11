#!/bin/bash
# ===========================================================================
# Human-specific ogbl-ppa link prediction (supervised fine-tuning)
# Uses proper Hydra override format (nested keys, no -- prefix)
# ===========================================================================

cd /oceanstor/home/e1553200/graph-gpt-main
source /oceanstor/home/e1553200/graphgpt311/bin/activate

# ---- Configurable parameters ----
MODEL_NAME="tiny"   # tiny|mini|small|medium|base
EPOCHS=2            # small for initial testing
BATCH_SIZE=128
NUM_WORKERS=4
LR=3e-4

# Model size lookup
case $MODEL_NAME in
  tiny)   HIDDEN=128; LAYERS=2 ;;
  mini)   HIDDEN=256; LAYERS=4 ;;
  small)  HIDDEN=512; LAYERS=4 ;;
  medium) HIDDEN=512; LAYERS=8 ;;
  base)   HIDDEN=768; LAYERS=12 ;;
  *)      HIDDEN=128; LAYERS=2 ;;
esac

WARMUP_EPOCHS=$(echo "scale=2; $EPOCHS * 0.3" | bc)
OUTPUT_DIR="./exp/models/ogbl_ppa_human/sv_h${HIDDEN}_l${LAYERS}_b${BATCH_SIZE}_e${EPOCHS}_lr${LR}"

echo "========================================"
echo "Human PPA Supervised Training"
echo "Model: $MODEL_NAME (h=$HIDDEN, l=$LAYERS)"
echo "Epochs: $EPOCHS, BS: $BATCH_SIZE, LR: $LR"
echo "Output: $OUTPUT_DIR"
echo "========================================"

python ./examples/train_supervised.py \
  tokenization.data.data_dir=./data/OGB \
  tokenization.data.data_path=ogbl_ppa_human \
  tokenization.data.dataset=ogbl-ppa-human \
  tokenization.tokenizer_class=StackedGSTTokenizer \
  'tokenization.data.sampling={edge_ego:{depth_neighbors:[[1,14]],neg_ratio:1,replace:false,method:{name:global},percent:100}}' \
  tokenization.attr_world_identifier=protein \
  tokenization.semantics.node.dim=2 \
  'tokenization.semantics.edge.discrete=null' \
  tokenization.semantics.edge.dim=0 \
  training.output_dir=$OUTPUT_DIR \
  'training.pretrain_cpt=' \
  training.task_type=edge \
  training.batch_size=$BATCH_SIZE \
  training.batch_size_eval=$BATCH_SIZE \
  training.num_workers=$NUM_WORKERS \
  training.schedule.epochs=$EPOCHS \
  training.schedule.warmup_epochs=$WARMUP_EPOCHS \
  training.optimizer.lr=$LR \
  training.optimizer.min_lr=0 \
  training.optimizer.eps=1e-10 \
  'training.optimizer.betas=[0.9,0.99]' \
  training.optimizer.weight_decay=0 \
  training.optimizer.max_grad_norm=1 \
  training.optimizer.use_ema=false \
  training.ft_eval.k_samplers=512 \
  training.ft_eval.epoch_per_eval=1 \
  training.ft_eval.eval_only=false \
  training.ft_eval.true_valid=-1 \
  training.ft_eval.save_pred=false \
  'training.deepspeed_conf_file=' \
  model.hidden_size=$HIDDEN \
  model.num_hidden_layers=$LAYERS \
  model.num_attention_heads=0 \
  model.intermediate_size=0 \
  model.hidden_act=gelu \
  model.max_position_embeddings=1024 \
  model.causal_attention=false \
  model.layer_scale_init_value=0 \
  model.graph_input.stack_method=short \
  model.graph_input.stacked_feat_agg_method=gated \
  model.dropout_settings.attention_dropout=0.1 \
  model.dropout_settings.path_dropout=0 \
  model.dropout_settings.embed_dropout=0 \
  model.dropout_settings.mlp_dropout=0 \
  model.ft_head.problem_type=single_label_classification \
  model.ft_head.num_labels=2

echo "Train and evaluation finished"
