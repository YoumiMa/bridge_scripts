# 環境整備（TSUBAME4.0）

NeMoのコンテナを獲得する

```bash

export APPTAINER_CACHEDIR=/gs/bs/tga-okazaki/ma/cache

apptainer build -s /gs/fs/tga-ma/ma/nemo-container docker:nvcr.io/nvidia/nemo:25.11
```

コンテナに入ってみる

```bash
# グループ領域をアクセスするために/gsをバインド
# module loadを可能とするために/appsをバインド
cd /gs/fs/tga-ma/ma/nemo-container
mkdir gs 
mkdir apps
touch usr/bin/nvidia-cuda-mps-server usr/bin/nvidia-smi usr/bin/nvidia-debugdump usr/bin/nvidia-persistenced usr/bin/nvidia-cuda-mps-control

apptainer shell -B /gs -B /apps -B /home -B /gs/fs/tga-okazaki/ma:/root -B /gs/bs/tga-okazaki/ma/cache:/root/.cache --nv -f -w /gs/fs/tga-ma/ma/nemo-container
```

# データ前処理

データは.jsonlか.jsonl.gz形式で、`"text"` というフィールドが含まれるようにする。

```json
{
  "text": "ゲオルク（ヨーラン）・ヴァーレンベリ（Georg (Göran) Wahlenberg、1780年10月1日 – 1851年3月22日）は、スウェーデンの博物学者である。カール・ツンベルク（トゥーンベリ）の後をついで、ウプサラ大学の博物学の教授を務めた。\n\nフィーリップスタードのKroppaで生まれた。1792年にウプサラ大学で医学の学位を得た。1806年に大学の職員に任命され、講師を務めた後、1814年にカール・ツンベルクの後をついで、植物学の教授に就任した。ヴァーレンベリはリンネ以来の、包括的な自然科学の教授職についた最後の人物で、自然科学の教授職は専門の分化が進んでいくことになり、植物学の分野にborgströmian教授職が設けられ、エリーアス・フリースが就任することになる。\n\n植物学の分野で、ラップランドの植物や高山植物の分布などの研究を行い、気象と植物の分布関係や分布限界にかんする論文を発表した。スウロバキアのハイタトラス山脈の植物などを科学調査した最初の科学者でもあり、Vyšné Wahlenbergovo湖やNižné Wahlenbergovo湖にヴァーレンベリの名前が残されている.。\n\n1808年にスウェーデン王立科学アカデミーの会員に選ばれた。キキョウ科のヒナギキョウ属（Wahlenbergia）やイグサ科の種、Luzula wahlenbergii に献名されている。\n\n著書\n Flora Carpatorum principalium exhibens plantas in montibus Carpaticis inter flumina Waagum et Dunajetz, cui praemittitur tractatus de altitudine, vegetatione, temperatura et meteoris horum montium in genere. Göttingen 1814.\n Flora Upsaliensis enumerans plantas circa Upsaliam sponte crescentes. Enchiridion excursionibus studiosorum Upsaliensium accomodatum. Upsaliae, R. Acad. Typographorum, 1820.\n",
  "meta": {
    "id": "2969838",
    "title": "ヨーラン・ヴァーレンベリ",
    "url": "https://ja.wikipedia.org/wiki/%E3%83%A8%E3%83%BC%E3%83%A9%E3%83%B3%E3%83%BB%E3%83%B4%E3%82%A1%E3%83%BC%E3%83%AC%E3%83%B3%E3%83%99%E3%83%AA"
  }
}
```

tokenizerのuse_fastの値を引数として渡せるように以下の編集を行う

```python
### opt/megatron-lm/tools/preprocess_data.py

def get_args():
		...

    group.add_argument('--keep-sequential-samples', action='store_true',
                       help='Ensure ordering of samples in .jsonl files is '
                            'preserved when using partitions>1.')
    # ---- add (begin) ----
    group = parser.add_argument_group(title='tokenizer options')
    group.add_argument('--use-fast-tokenizer', action='store_true',
                       help='Use fast tokenizer (default: False)')
    # ---- add (end) ----
                       
    ...
                       
    args.rank = 1
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0

　　# add below
    args.use_fast = args.use_fast_tokenizer if hasattr(args, 'use_fast_tokenizer') else False

### opt/megatron-lm/megatron/core/tokenizers/text/utils/build_tokenizer.py

def build_tokenizer(args):
...
    elif args.tokenizer_type == 'HuggingFaceTokenizer':
        tokenizer_library = 'huggingface'
        tokenizer_path = args.tokenizer_model
        kwargs['vocab_file'] = args.vocab_file
        kwargs['merges_file'] = args.merge_file
        # add below
        kwargs['use_fast'] = args.use_fast

```

.jsonl.gz形式でも処理できるように以下の編集を行う

```python
### opt/megatron-lm/tools/preprocess_data.py

def process_json_file(self, file_name):
        input_file_name, output_prefix = file_name
        print("Opening", input_file_name)
        # ---- add (begin) ---- 
        if input_file_name.endswith(".gz"):
            fin = gzip.open(input_file_name, 'rt', encoding='utf-8')
        else:
            fin = open(input_file_name, 'r', encoding='utf-8')
        # ---- add (end) ---- 
        # comment out below
        # fin = open(input_file_name, 'r', encoding='utf-8')

        startup_start = time.time()	
```

さらに、処理したトークン数を表示するように以下の編集を行う

```python
### opt/megatron-lm/tools/preprocess_data.py

def process_json_file(self, file_name):

...

        startup_end = time.time()
        proc_start = time.time()
        total_bytes_processed = 0
        total_tokens_processed = 0  # add this line
	      ...
        for key in doc.keys():
            total_tokens_processed += len(doc[key])  # add this line
            builders[key].add_document(doc[key], sentence_lens[key])
        self.print_processing_stats(i, proc_start, total_bytes_processed, total_tokens_processed) # modify this line

def print_processing_stats(self, count, proc_start, total_bytes_processed, total_tokens_processed = 0): # add total_tokens_processed = 0
	if count % self.args.log_interval == 0:
		current = time.time()
		elapsed = current - proc_start
		mbs = total_bytes_processed/elapsed/1024/1024
		tps = total_tokens_processed/elapsed  ## add this line
		print(f"Processed {count} documents",
		f"({count/elapsed:.2f} docs/s, {mbs:.2f} MB/s, {total_tokens_processed:,} tokens, {tps:.2f} tokens/s).", # add {total_tokens_processed:,} tokens,
		file=sys.stderr)
        
        
```

最後に、以下のスクリプトでデータのトークナイゼーションを行う

```bash
#!/bin/bash

# テキストファイルのパス
OUTPUT_ROOT="/path/to/output/dir"
mkdir -p $OUTPUT_ROOT

BASE_PATH="/path/to/input/dir"
CONTAINER_IMAGE="/path/to/container"

MEGATRON_LM_PATH="/opt/megatron-lm/"

for DIR in "$BASE_PATH"/; do
    FILE_PATHS=("${DIR}"*)
    echo "Processing directory: $DIR"
    # FILE_PATHSを使った処理
    for FILE_PATH in "${FILE_PATHS[@]}"; do
        echo "  Processing $FILE_PATH"
        FILE_NAME=$(basename "$FILE_PATH")
        OUTPUT_PREFIX="${OUTPUT_ROOT}/${FILE_NAME%.jsonl.gz}"

        echo $OUTPUT_PREFIX
        apptainer run --nv \
  -w -f -B /gs -B /apps -B /home -B /gs/fs/tga-okazaki/ma:/root ${CONTAINER_IMAGE} \
        python ${MEGATRON_LM_PATH}/tools/preprocess_data.py \
            --input "$FILE_PATH" \
            --output-prefix "$OUTPUT_PREFIX" \
            --tokenizer-type HuggingFaceTokenizer \
            --tokenizer-model tokyotech-llm/Llama-3.1-Swallow-8B-v0.5 \
            --append-eod \
            --use-fast-tokenizer \
            --workers 64
        sleep 1
    done
done
```

# HF → Megatron チェックポイント変換

以下のスクリプトでHFチェックポイントをMegatron形式に変換する

```bash
HF_MODEL="tokyotech-llm/Llama-3.1-Swallow-8B-v0.5"
MEGATRON_PATH="/gs/bs/tga-ma/ma/ckpts/llama-3.1-swallow-8B-v0.5"

CONTAINER_IMAGE="/gs/fs/tga-ma/ma/nemo-container"
MEGATRON_BRIDGE_PATH="/opt/Megatron-Bridge/"

apptainer run --nv \
  -w -f -B /gs -B /apps -B /home -B /gs/fs/tga-okazaki/ma:/root ${CONTAINER_IMAGE} \
  python ${MEGATRON_BRIDGE_PATH}/examples/conversion/convert_checkpoints.py import \
  --hf-model ${HF_MODEL} \
  --megatron-path ${MEGATRON_PATH}
```

# 継続事前学習

Megatron-Bridgeではハイパーパラメータをyamlファイルで管理し、python scriptでそれを呼び出すようにできる。また、以下のように実装するとCLIでハイパラを上書きできる。
[参考](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/models/qwen3_next/finetune_qwen3_next_80b_a3b.py)

Megatron-Bridge内装のレシピ（recipes/llama/llama3.pyなど）そのまま使うと、モデルの重みがランダムに初期化される（ゼロからの事前学習になる）ので、重みを読み込めるようにする。
```python
## /opt/Megatron-Bridge/src/megatron/bridge/recipes/llama/llama3.py

def _llama3_common(
    hf_path: str,
    dir: str | None = None,
    name: str = "default",
    ...
    comm_overlap_config: CommOverlapConfig | None = None,
    load_weights: bool = True, ## add this line
    ) -> ConfigContainer:
    
    ...
    bridge = AutoBridge.from_hf_pretrained(hf_path)
    # model_cfg = bridge.to_megatron_provider(load_weights=False) ## comment out this line
    model_cfg = bridge.to_megatron_provider(load_weights=load_weights) ## add this line
```

以下のスクリプトにより学習を実行できる。

```
bash qsub_merged_run_exp1.sh
```

ただし`$PE_HOSTFILE, $NHOSTS`などの環境変数は、ローカル環境に依存する可能性がある。
なお、マルチノード化せず、シングルノードで学習する場合、$MASTER_ADDR, $MASTER_PORTなどの環境変数を指定せず、
torchrunで学習できると思われる。具体的には、[公式のチュートリアルページ](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/tutorials/recipes/llama)に参照されたい。



# Megatron → HF チェックポイント変換

以下のスクリプトでMegatronチェックポイントをHF形式に変換する

```bash
HF_MODEL="tokyotech-llm/Llama-3.1-Swallow-8B-v0.5"
MEGATRON_PATH="/gs/bs/tga-ma/ma/ckpts/llama-3.1-swallow-8B-v0.5"
HF_PATH="/gs/bs/tga-ma/ma/ckpts/llama-3.1-swallow-8B-v0.5/hf"
CONTAINER_IMAGE="/gs/fs/tga-ma/ma/nemo-container"
MEGATRON_BRIDGE_PATH="/opt/Megatron-Bridge/"

apptainer run --nv \
  -w -f -B /gs -B /apps -B /home -B /gs/fs/tga-okazaki/ma:/root ${CONTAINER_IMAGE} \
  python ${MEGATRON_BRIDGE_PATH}/examples/conversion/convert_checkpoints.py export \
  --hf-model ${HF_MODEL} \
  --megatron-path ${MEGATRON_PATH} \
  --hf-path ${HF_PATH}
```
