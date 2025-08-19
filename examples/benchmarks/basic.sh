SCENE_DIR="../../360_v2"
RESULT_DIR="../../results/gsplat"
SCENE_LIST="garden bicycle stump bonsai counter kitchen room" # treehill flowers
RENDER_TRAJ_PATH="ellipse"
POSTIFX_LIST = "_retinex_contrast _retinex_multiexposure _retinex_variance"

for POSTFIX in $POSTIFX_LIST;
  for SCENE in $SCENE_LIST;
  do
      if [ "$SCENE" = "bonsai" ] || [ "$SCENE" = "counter" ] || [ "$SCENE" = "kitchen" ] || [ "$SCENE" = "room" ]; then
          DATA_FACTOR=2
      else
          DATA_FACTOR=4
      fi

      echo "Running $SCENE"

      # train without eval
      CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default --disable_viewer --data_factor $DATA_FACTOR \
          --render_traj_path $RENDER_TRAJ_PATH \
          --postfix $POSTFIX \
          --data_dir ../../360_v2/$SCENE/ \
          --result_dir $RESULT_DIR/$POSTFIX/$SCENE/
  done
done