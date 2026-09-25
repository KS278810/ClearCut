# モデル変換/取得手順

`models/`はgitignore対象。各モデルは以下の手順で用意する。

## YOLOX-S — 変換不要(Phase B)

`18_背景除去/checkpoints/yolox_s.onnx`をそのままコピーする。ONNX Runtime Webは通常のONNXファイルをそのまま読み込めるため、変換作業は不要(要動作確認 — opsetやオペレータがWASM/WebGPU EPでサポートされているかはPhase Bで確認する)。

```
cp ../18_背景除去/checkpoints/yolox_s.onnx models/
```

## BiRefNet_lite — 512x512に自前再エクスポートが必要(実測済み・完了)

18_背景除去が使う `checkpoints/birefnet_lite.onnx`(1024x1024固定入力)は、**そのままではブラウザで動作しない**ことが実測で判明した:

- **WASM実行**: `onnxruntime-node`で実際に推論しメモリ使用量を計測したところ、1024x1024の1推論で**約5.1GBの活性化メモリ**を消費する。WASMは32bitアドレス空間のため最大4GBまでしか確保できず、`std::bad_alloc`で失敗する(このプロジェクトのブラウザ実機テストで再現確認済み)。fp16版(`onnx-community/BiRefNet_lite-ONNX`の`model_fp16.onnx`)も試したが、CPU実行時にfp16→fp32変換のオーバーヘッドが乗り、**むしろ7.7GBに悪化**した。
- **WebGPU実行**: `microsoft/onnxruntime#21968` — Swinバックボーンが生成するConcat/Splitのストレージバッファバインド数が上限(このプロジェクトの環境では16)を超過し、"Too many storage buffers in shader"で実行時エラー。2026年6月時点でも未解決の既知issue。解像度を下げても、このアーキテクチャ由来の問題は解消しない。

**採用した対応**: `ZhengPeng7/BiRefNet_lite`のsafetensorsから、入力解像度512x512で自前ONNXエクスポートした`birefnet_lite_512.onnx`をWASM実行で使う。512x512版の推論メモリは**約1.95GB**(実測)で、WASMの4GB上限に収まる。WebGPUは512版でも上記のConcat/Split問題が残るため、依然WASM固定。

エクスポート手順(torch>=2.0のexporterはこのモデルの`ASPPDeformable`ブロックに対応していないため、`deform_conv2d_onnx_exporter`パッチ+torch 1.13.1が必要 — 18_背景除去の`docs/DECISIONS.md`のBiRefNet_lite-mattingエクスポート作業と同じ制約):

```
# 短いパスに一時venvを作成(Windowsのパス長制限を回避)
python -m venv D:/tmp_bnexp512
D:/tmp_bnexp512/Scripts/pip install torch==1.13.1+cpu torchvision==0.14.1+cpu --index-url https://download.pytorch.org/whl/cpu
D:/tmp_bnexp512/Scripts/pip install timm einops "transformers<4.40" safetensors huggingface_hub onnx onnxscript opencv-python-headless pillow numpy
D:/tmp_bnexp512/Scripts/pip install "kornia==0.6.12" --no-deps  # 新しいkorniaはtorch>=2.0必須、torch1.13を巻き込みアップグレードしてしまうため--no-deps必須

# ZhengPeng7/BiRefNet_lite の birefnet.py / BiRefNet_config.py / config.json / model.safetensors を取得
# birefnet.py内の `from .BiRefNet_config import BiRefNetConfig` を `from BiRefNet_config import BiRefNetConfig` に書き換え(相対import解消)
# deform_conv2d_onnx_exporter.py の get_tensor_dim_size() にNoneフォールバックパッチを適用(masamitsu-murase/deform_conv2d_onnx_exporter)

# export512.py: torch.randn(1,3,512,512)でトレース、opset_version=17、
# net(x)[-1]でラップして単一出力(sigmoid前logit)のみexport
```

生成物のシャニティチェック(Node.js、`onnxruntime-node`):
```js
const session = await ort.InferenceSession.create('birefnet_lite_512.onnx', { executionProviders: ['cpu'] });
const tensor = new ort.Tensor('float32', new Float32Array(1*3*512*512), [1,3,512,512]);
const results = await session.run({ [session.inputNames[0]]: tensor });
console.log(process.memoryUsage().rss / 1e9, 'GB'); // ~1.95 GB
```

`models/birefnet_lite_512.onnx`として既に配置済み(このリポジトリの`models/`はgitignore対象なので、再現する場合は上記手順で再エクスポートするか、この会話のエクスポート成果物を別途共有する必要がある)。

## SAM2 Hiera-Tiny — ONNX化が必要(Phase D、未着手)

18_背景除去では torch経由でサブプロセス実行しており、ONNXエクスポートはこのプロジェクトで前例がない。Phase Dで以下の順に検討する:

1. まず既存のコミュニティ変換を試す: `SharpAI/sam2-hiera-tiny-onnx`(Hugging Face)。エンコーダ/デコーダが分離した形で提供されている。精度・動作が十分ならこれを採用。
2. 不十分な場合、`sam2`パッケージ(Apache-2.0)から自前でONNXエクスポート(画像エンコーダ+プロンプトエンコーダ+マスクデコーダ)。参考実装: `lucasgelfond/webgpu-sam2`。

TODO: Phase D着手時に実際に変換・動作確認し、この節を更新する。
