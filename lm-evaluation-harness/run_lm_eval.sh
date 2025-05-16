strings=(
  "shengyuanhu/wmdp_unlearn_rmu_150_1200_6.5_zephyr"
)

for path in "${strings[@]}";
do
lm-eval --model hf \
    --model_args pretrained=$path,tokenizer="HuggingFaceH4/zephyr-7b-beta" \
    --tasks mmlu \
    --batch_size=16 \
    --device cuda:1
done