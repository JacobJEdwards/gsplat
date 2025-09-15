SCENE_DIR="/workspace/360_v2"
RESULT_DIR="/workspace/results/gsplat"
SCENE_LIST=(
  "garden"
  "bicycle"
  "stump"
  "bonsai"
  "counter"
  "kitchen"
  "room"
)
RENDER_TRAJ_PATH="ellipse"
POSTFIX_FIRST_PART=(
  "_retinex_contrast"
  "_retinex_multiexposure"
  "_retinex_variance"
)
#POSTFIX_SECOND_PART=(
#  "_retinexmamba_LOL_v2_synthetic"
#  "_retinexmamba_LOL_v2_real"
#  "_retinexformer_SMID"
#  "_retinexformer_SID"
#  "_retinexformer_SDSD_outdoor"
#  "_retinexformer_SDSD_indoor"
#  "_retinexformer_LOL_v2_synthetic"
#  "_retinexformer_LOL_v2_real"
#  "_retinexformer_LOL_v1"
#  "_retinexformer_FiveK"
#)

POSTFIX_LIST=""
for FIRST in "${POSTFIX_FIRST_PART[@]}";
do
  for SECOND in "${POSTFIX_SECOND_PART[@]}";
  do
    POSTFIX_LIST+="${FIRST}${SECOND} "
  done
done

for POSTFIX in $POSTFIX_LIST;
do
  for SCENE in "${SCENE_LIST[@]}";
  do
      echo "Running $SCENE on $POSTFIX"
      DATA_DIR="$SCENE_DIR/$SCENE"

      # train without eval
      CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default --disable_viewer \
          --render_traj_path $RENDER_TRAJ_PATH \
          --postfix $POSTFIX \
          --data_dir $DATA_DIR \
          --result_dir $RESULT_DIR/$POSTFIX/$SCENE/
  done
done