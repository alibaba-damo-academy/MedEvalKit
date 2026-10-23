#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
# MMMU-Medical-test,MMMU-Medical-val,PMC_VQA,MedQA_USMLE,MedMCQA,PubMedQA,OmniMedVQA,Medbullets_op4,Medbullets_op5,MedXpertQA-Text,MedXpertQA-MM,SuperGPQA,HealthBench,IU_XRAY,CheXpert_Plus,MIMIC_CXR,CMB,CMExam,CMMLU,MedQA_MCMLE,VQA_RAD,SLAKE,PATH_VQA,MedFrameQA
EVAL_DATASETS="Medbullets_op4" 
DATASETS_PATH="hf"
OUTPUT_PATH="eval_results/{}"
# TestModel,Qwen2-VL,Qwen2.5-VL,BiMediX2,LLava_Med,Huatuo,InternVL,Llama-3.2,LLava,Janus,HealthGPT,BiomedGPT,Vllm_Text,MedGemma,Med_Flamingo,MedDr
MODEL_NAME="Qwen2.5-VL"
MODEL_PATH="Qwen2.5-VL-7B-Instruct"

#vllm setting
CUDA_VISIBLE_DEVICES="0,1,2,3"
USE_VLLM="False"
IFS=',' read -r -a GPULIST <<< "$CUDA_VISIBLE_DEVICES"
TOTAL_GPUS=${#GPULIST[@]}
CHUNKS=${CHUNKS:-$TOTAL_GPUS} 

#Eval setting
SEED=42
REASONING="False"
TEST_TIMES=1


# Eval LLM setting
MAX_NEW_TOKENS=8192
MAX_IMAGE_NUM=6
TEMPERATURE=0
TOP_P=0.0001
REPETITION_PENALTY=1

# LLM judge setting
USE_LLM_JUDGE="True"
# gpt api model name
GPT_MODEL="gpt-4.1-2025-04-14"
OPENAI_API_KEY=""


for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python eval.py \
    --eval_datasets "$EVAL_DATASETS" \
    --datasets_path "$DATASETS_PATH" \
    --output_path "$OUTPUT_PATH" \
    --model_name "$MODEL_NAME" \
    --model_path "$MODEL_PATH" \
    --seed $SEED \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --max_image_num "$MAX_IMAGE_NUM" \
    --use_vllm "$USE_VLLM" \
    --num_chunks $CHUNKS \
    --chunk_idx $IDX \
    --reasoning $REASONING \
    --temperature "$TEMPERATURE"  \
    --top_p "$TOP_P" \
    --repetition_penalty "$REPETITION_PENALTY" \
    --use_llm_judge "$USE_LLM_JUDGE" \
    --judge_model "$JUDGE_GPT_MODEL" \
    --api_key "$OPENAI_API_KEY" \
    --test_times "$TEST_TIMES"  &  # 这里的反斜杠后面不要有空格
done


wait
