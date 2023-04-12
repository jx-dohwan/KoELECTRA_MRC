
## 💡프로젝트 소개

#### 1️⃣ 주제 : KoELECTRA를 활용한 Q&A를 위한 기계독해<br>
#### 2️⃣ 설명 : [KorQuAD: 기계독해를 위한 한국어 질의응답 데이터셋](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE07613668)을 기반으로 기계독해 모델 구현<br> 
#### 3️⃣ 모델 : Hugging Face [monologg/koelectra-base-v3-discriminator](https://huggingface.co/monologg/koelectra-base-v3-discriminator) 모델 사용하여 진행<br><br>

### 해당 프로젝트에 관한 자세한 사항은 블로그에 정리해 놓았다.
- [KoELECTRA를 활용한 Q&A를 위한 기계독해_1(ft.베이스라인 이론편)](https://velog.io/@jx7789/KoELECTRA%EB%A5%BC-%ED%99%9C%EC%9A%A9%ED%95%9C-QA%EB%A5%BC-%EC%9C%84%ED%95%9C-%EA%B8%B0%EA%B3%84%EB%8F%85%ED%95%B41ft.%EB%B2%A0%EC%9D%B4%EC%8A%A4%EB%9D%BC%EC%9D%B8%ED%8E%B8)
- [KoELECTRA를 활용한 Q&A를 위한 기계독해_2(ft.베이스라인 코드편)](https://velog.io/@jx7789/KoELECTRA%EB%A5%BC-%ED%99%9C%EC%9A%A9%ED%95%9C-QA%EB%A5%BC-%EC%9C%84%ED%95%9C-%EA%B8%B0%EA%B3%84%EB%8F%85%ED%95%B41ft.%EB%B2%A0%EC%9D%B4%EC%8A%A4%EB%9D%BC%EC%9D%B8-%EC%BD%94%EB%93%9C%ED%8E%B8)


## 논문 소개
- 한국어 위키백과를 기반으로 한 대규모 기계 독해 데이터셋으로 KorquAD 1.0~2.0이 있지만 여기서는 1.0 버전을 활용한다. 
<br>

![](img/korquad.png)
### 부연설명
- 스탠포드 대학교의 SQuAD 1.0를 표방한 데이터셋
- 1,560개의 한국어 위키피디아 문서에서 10,645건의 문단과 66,181개의 질의응답 쌍
- Training set 60,407 / Dev set 5,774 질의응답 쌍으로 구성


---
## 1. train

```
!python run_korquad.py \
    --model_type electra \
    --model_name_or_path monologg/koelectra-base-v3-discriminator \
    --output_dir koelectra-base-v3-korquad \
    --data_dir data \
    --train_file korquad/KorQuAD_v1.0_train.json \
    --predict_file korquad/KorQuAD_v1.0_dev.json \
    --max_seq_length 512 \
    --doc_stride 128 \
    --max_query_length 64 \
    --do_train \
    --do_eval \
    --evaluate_during_training \
    --per_gpu_train_batch_size=16 \
    --per_gpu_eval_batch_size=8 \
    --learning_rate 5e-5 \
    --gradient_accumulation_steps 1 \
    --weight_decay 0.0 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 1.0 \
    --num_train_epochs 3 \
    --max_steps -1 \
    --warmup_steps 0 \
    --n_best_size 20 \
    --max_answer_length 30 \
    --verbose_logging \
    --logging_steps 1000 \
    --save_steps 1000 \
    --eval_all_checkpoints \
    --overwrite_output_dir \
    --seed 42 \
    --local_rank -1 \
    --threads 4
```

## 2. Test
```
from evaluate import evaluate_result

dataset_file = 'data/korquad/KorQuAD_v1.0_dev.json'
prediction_file = 'koelectra-base-v3-korquad/predictions_korquad.json'
evaluate_result(dataset_file, prediction_file)
```

```
from evaluate import analyze_result

dataset_file = 'data/korquad/KorQuAD_v1.0_dev.json'
prediction_file = 'koelectra-base-v3-korquad/predictions_korquad.json'
f1_threshold = 0.85
model_name_or_path = 'koelectra-base-v3-korquad/checkpoint-12000'
analyze_result(dataset_file, prediction_file, f1_threshold, model_name_or_path)
```

---
## 🗓️ 프로젝트 개선 진행

|개선사항|기타|진행률(%)|진행할 사항|
|:-----:|:-----:|:-----:|:-----:|
|2.0으로 업그레이드|AIHub데이터, KorQuAD 2.0 사용|||
|다중 태스크 학습|다양한 태스크의 데이터를 학습하여 효율성 극대화|||

---
