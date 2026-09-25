# Decision log

Research findings and adopt/reject verdicts that don't belong in the
user-facing README, kept here so a future session doesn't have to re-derive
them from scratch. Newest entries at the top of each section.

## Working discipline (read before pre-registering a threshold)

Added 2026-09-24 after keep_main_subject's SAM2 refinement failed its gate
twice in a row (第11計画 → 第12計画) for reasons that were both visible in
data already on hand before the gate ran. The pattern both times: decide a
threshold/rule, register it, run the full production gate (GPU, tens of
minutes), THEN discover from the failure that the number itself was the bug
-- when the evidence to see that was already sitting in the investigation
notes or a two-line calculation. See 第13計画's Adopted entry for the full
なぜなぜ that led here.

- **Before registering a threshold that targets specific known-failure
  frames, calibrate against those exact frames first** (or their cached
  intermediate data -- no re-render needed) instead of registering a number
  and finding out from the production gate whether it was right. A design
  choice earns its pre-registration by surviving a calibration pass, not by
  sounding reasonable.
- **A threshold needs a derivation, not just a value.** If you can't say in
  one sentence why THIS number and not a nearby one, that's the sign to
  measure a distribution first (e.g. keep_main_subject_sam2_overlap=0.30
  had one -- "person fragments ~1.0, props ~0.0, huge margin"; the box-
  area-ratio floor that broke everything did not).
- **A safety net you added because a failure mode seems plausible is itself
  a hypothesis, not a fact -- measure whether it ever fires and what it
  rejects before shipping it.** The min_mask_frac floor was added purely as
  defensive plausibility ("SAM2 might lose the subject") and turned out to
  be the ONLY thing that ever fired, always on exactly the frames the lever
  existed to fix, always rejecting a mask that was in fact correct. This
  project's oldest and best-documented lesson (v23→v24's deleted
  heuristics) is that an unmeasured defensive addition is a bug waiting to
  be discovered, not free insurance.
- **"Don't re-tune a pre-registered threshold after seeing the gate fail"
  still applies** -- it exists to stop p-hacking a threshold into passing.
  It does not mean freezing a gate that failed because of a genuinely wrong
  mechanism forever; the correct move is a NEW, independently pre-registered
  attempt with a derivation this time, not silently nudging the old number.

## Fixture / dataset changes

### sample/flatchroma2/ added -- flat-chroma coverage restored (2026-09-14)

Four clips (768x768, 124 frames, blue `#394ABF` key colour, from the
`01_動画生成` render batch of 2026-09-12). All four gate as
`bg_is_chroma_class=True` (`frac_bg_like` 0.9984-1.0000).

This closes the gap the 2026-09-13 reorg opened and that `tool/tests/
test_regression_gates.py` explicitly flagged: after that reorg `sample/` held
NO flat-chroma clip, so the F1i gate -- the one guarding the pipeline's
original design target and its worst historical defect class -- had nothing to
run against and skipped itself. Wired into `tool/qc/fixtures.py` as the
`"flatchroma2"` set.

### sample/dinosaur/ reduced to a single clip (2026-09-13)

The user reorganised `01_背景除去/sample/` (the sibling directory holding
every project's raw source clips) directly, concurrently with an
in-progress Claude Code session. As part of that reorg, the original four
`sample/dinosaur/` clips (思考/喜び/感謝/挨拶.mp4 -- a flat-yellow-chroma
3D-rendered dinosaur mascot, the subject of most of this log's `V0`-`V7`/
`Phase 0-4` entries and the "18-bg-remove plan" this repo's regression
gates reference) were deleted as no longer needed, leaving a single
different clip, `Triceratops.mp4`, in their place. Two real-footage
fixture directories were also renamed in the same reorg (see the
"Fixture sets reduced to dinosaur-only" entry below for their current
status -- both were removed from this repo's published tree on
2026-09-25).

**Important difference discovered while re-pointing tests**:
`Triceratops.mp4` is NOT flat-chroma (`server.probe.validate_upload`
measures `bg_is_chroma_class=False`, `bg_frac_bg_like=0.943 < 0.98`) --
unlike the four clips it replaced. `tool/tests/test_regression_gates.py`'s
F1i gate (chroma-distance-based, `None` for a non-chroma clip per
`qc.metrics.evaluate`) now skips rather than asserts for this clip; the
S2/S3 gates (colour-freeze/GIF-flicker regressions, chroma-independent)
still run normally. `results_dinosaur/` was regenerated against
`Triceratops.mp4` with the current default `PipelineConfig` (device=cuda)
to replace the four now-orphaned GIFs. Every other reference to the old
four clips (`tool/qc/fixtures.py`, `tool/scripts/speed_experiment.py`,
`tool/tests/test_runner_limits.py`, `server/tests/test_probe.py`,
`server/static/tests/e2e.test.mjs`) was updated to `Triceratops.mp4` (121
frames, not 122).

Net effect: this repo's automated regression coverage for the pipeline's
original flat-chroma design target (F1i specifically) is weaker than
before this reorg -- there is currently no flat-chroma clip left in
`sample/` for that one gate to exercise. Revisit if/when a flat-chroma
sample becomes available again.

### Fixture sets reduced to dinosaur-only; other clip identifiers generalised throughout (2026-09-25)

Per a decision to keep only this project's own `sample/dinosaur/` material
in the published tree, every other fixture set was removed:

- The `"flatchroma2"` set (the 2026-09-14 flat-chroma restoration above)
  and its `results_flatchroma2/` deliverable
- Two real-footage regression sets (`cyclorama`, a white-cyclorama batch;
  `purplebg`, a client-adjacent purple-backdrop batch), and their
  `results_cyclorama/`/`results_purplebg/` deliverables
- A wide-pose real-footage fixture (`widepose`) used to investigate/fix
  the `keep_main_subject` prop-removal defect (第11-13計画)

`tool/qc/fixtures.py` now defines only the `"chroma"` (dinosaur) set;
`tool/tests/test_regression_gates.py`'s flatchroma2-based F1i/F2 gate was
removed (F1i/F2 go back to skipping for the natural-background
Triceratops.mp4, same as before the 2026-09-14 restoration -- see that
entry above). Four one-off experiment scripts that depended on the
removed clips (`box_mode_experiment.py`, `sam2_trimap_prototype.py`,
`sam2_keep_main_subject_experiment.py`, `speed_experiment.py`) were
deleted -- their measured results are already recorded in this log's
relevant entries and are not re-derivable by re-running them without
that material.

Every reference to the original clip names elsewhere in this codebase
(code comments, docstrings, and this log's own entries below) was
replaced with the bare identifiers above (`flatchroma2`/`cyclorama`/
`purplebg`/`widepose`) without changing any measured number or
conclusion -- they now read as this repo's own fixture nicknames rather
than the original directory names.

**Coverage lost**: the flat-chroma F1i/F2/S2/S3 regression gate (the
pipeline's single most important defect class, see this log's `v23`->`v24`
history) currently has no flat-chroma fixture to run against outside of
`tool/tests/test_keyer.py`'s synthetic frames. If this project's own
flat-chroma material (not a client deliverable) becomes available again,
re-add it to `tool/qc/fixtures.py` as a new named set and restore the
corresponding gate in `test_regression_gates.py` (see the removed
`flatchroma2` code in git history for the pattern).

## Adopted

### keep_main_subject規則(iii)のSAM2画素マスク化(第13計画) — opt-inとして採用、既定OFF維持(2026-09-24)

第12計画で不合格になったSAM2画素マスク化(上記「Researched but not adopted」の
同名エントリ参照)を、なぜなぜ分析のうえ再挑戦し成功させた。

**なぜなぜ分析の要点**: 第11計画・第12計画とも「本番ゲートを回してから」欠陥に
気づいたが、どちらの欠陥も着手前の手元データで予見できるものだった。第12計画では
「人物成分≈28.3k px、箱≈156k px→比率0.18」が調査報告の時点で既に見えていたのに、
`min_mask_frac=0.20`という導出なしの数値を確認せずそのまま本番ゲートに投入していた。
根本原因は「測る前に閾値を決める」というプロセスの欠落であり、SAM2という技術方針
自体は正しかった(dinosaur/flatchroma2で副作用ゼロと既に実証済み)。

**Phase 0(校正、レンダ無し、`--calibrate`)**: 6クリップ・90決定点フレーム・
小成分143個の分布を実測。wideposeの実際の小道具(ダンベル/ボトル/ノートPC、
area 500〜4600px)62個中54個はSAM2人物マスクとの重なり率が**完全に0.000**、
非ゼロの8個は全てarea<70pxの人物本体の微小断片で重なり率1.000。空マスクは
90決定点全てでゼロ件。フォールバックしたフレーム`[105-110,177-182]`は小道具
出現区間と完全一致——健全性チェック(`min_mask_frac=0.20`)は「まさに直したい
フレーム」だけを、開脚姿勢で箱に対する人物シルエット比率が構造的に小さくなる
という理由で狙い撃ちして誤動作していた。

**設計変更**: `keep_main_subject_sam2_min_mask_frac`(下限0.20/上限2.0)を撤廃。
`_component_keep_labels`の判定を「SAM2マスクが空でない限り採用」に単純化
(`tool/pipeline/stages.py`)。overlap閾値0.30・dilate_px=5は校正データで
検算済みで変更不要。あわせて`keep_main_subject`自体(規則(i)(ii)含む全体)を
`runner.run_clip`でkeyer経路に対し完全無効化(第12計画時点ではSAM2部分の
`sam2_fn`だけがkeyer経路で無効化されており、規則(i)(ii)自体はkeyer経路でも
動いてF1i +1.2〜9.7%悪化(下記の第12計画エントリ参照)を引き起こしていた
——見落としの修正)。

**dry-run(本番ゲート前の事前検算)**: wideposeの小道具出現区間46フレーム全てを
キャッシュ済みrawで処理(レンダ無し)。SAM2フォールバック0/23。frame 110/180を
含む区間全体でダンベル・ボトル・ノートPCは完全に消えた。残存した4フレームの
1〜69px微小断片は全て重なり率1.000(人物本体の一部として正しく保護)——本レバー
とは無関係な既存ノイズと個別確認済み。

**本番ゲート実測(6クリップ、`tool/scripts/sam2_keep_main_subject_experiment.py`)**:

| 指標 | 結果 |
|---|---|
| widepose frame 110 | `[28332,2863,1147,1117,37,9]` → `[28332]`(単一成分、完全一致) |
| widepose frame 180 | `[27561,2542,1140,989]` → `[27561]`(単一成分、完全一致) |
| widepose F1i/S1 | 212377/685.4 → 214264/582.9(F1i微増だがS1改善、小道具削除の副作用は許容範囲) |
| widepose S2(色ちらつき) | 59.6 → 41.1(-31%、副次的改善) |
| dinosaur F1i/S1 | null(非クロマ)/8125.2→8123.5(実質不変) |
| flatchroma2_c1/思考/挨拶 F1i/S1 | 全て完全一致 |
| flatchroma2_c2 F1i/S1 | F1i完全一致、S1誤差レベル(2505.268→2505.293) |
| postprocess_s増分 | 全クリップ+28.5〜49.5s(基準60s以内) |
| SAM2フォールバック率 | **90決定点中0件=0%**(基準1%を大幅クリア) |

事前登録した4ゲート(widepose主目標・非退行・時間増分・フォールバック率)は
**全て合格**。

**既定反転は見送り**: 計画の事前登録どおり、既定ON化には汎化クリップ
(widepose以外の「床に小道具、ニューラル経路」素材)での確認が条件だが、
ユーザー決定(2026-09-24)により**汎化クリップは用意できない・後回し**。
したがって`keep_main_subject`は**既定Falseのまま**、`--keep-main-subject`
CLIフラグ/API override `keep_main_subject: true`での**opt-in**として出荷する。
widepose型(灰色・自然背景でニューラル経路、床に小道具)の素材では有効性が
実証済みだが、汎化は未測定。汎化クリップが用意でき次第、本エントリのゲート2
(未実施)を開封して既定ON化を再判定すること。

コミット: `b5fa68d`(健全性チェック撤廃+keyer無効化、pytest 26 passed/対象
ファイル)。`data/keep_main_subject_sam2_experiment/`と
`data/keep_main_subject_sam2_calibration/`(計約110MB)はこの記録に転記済みの
ため削除可。


### 第11計画 Part 1: 速度と応答性 -- seek廃止・per_frame 1ループ復帰・prepare段・ETAフレーム比例化 (adopted 2026-09-23)

実機フィードバック(「dinosaur GPUが7分と出る」「最初の1枚が出るまで無反応」「purplebg 挨拶が3分」)の
調査で特定した原因への対処。すべて出力を変えない(1-2は41b3591以前の出力へ戻す)変更。

**1-1 キー判定サンプリングのseek廃止**(commit 3b46cb0): `_sample_frames_for_key`の24回の
`cap.set(CAP_PROP_POS_FRAMES)`は、キーフレームが先頭1つだけのサンプル素材では毎回先頭から
デコードし直していた。`grab()`で順送りし対象インデックスだけ`retrieve()`する。抽出フレームは
旧実装と完全一致(テストで旧実装をオラクルとして固定)。実測(load average 約10、同一負荷で前後):

| clip | 旧(seek) | 新(grab) | 抽出結果 |
|---|---|---|---|
| dinosaur/Triceratops.mp4 (1656x1248, 121f) | 19.9s | 3.0s | 完全一致 |
| purplebg/挨拶.mp4 (1440², 122f) | 11.8s | 1.0s | 完全一致 |
| widepose/widepose.mp4 (960x540, 245f) | 12.7s | 1.0s | 完全一致 |

調査時(load average 30〜37)の55〜70sより小さいのは負荷差。全経路(keyerに拒否されるクリップも含む)で効く。

**1-2 per_frameを1ループに戻す / 1-3 prepare段**(commit 430d2a9): レバーC(不採用)の2パス化
(41b3591)で、既定のper_frameでも全フレームのデコード→YOLOX→BiRefNetが終わるまで最初の
preview/progressが出なくなっていた。per_frameは フレームごとに デコード→YOLOX→BiRefNet(切り落とし時は
単一フレーム全画面再推論`auto_full_frame_fallback=True`)→despill→preview/progress に戻し、2パスは
opt-inの`clip_union`/`smoothed`専用に残した。`infer_clip`冒頭で`progress("prepare",0,1)`、
`server/jobs.py`の`STAGES`に`prepare`(重み0.02)を追加し、ハートビートが開始直後から送信される。
Triceratops GPU(Modelsを未ロードから、load average 12〜29):

| | 旧(b19617c) | 新 |
|---|---|---|
| 最初のprogress | 93.0s (infer) | 0.0001s (prepare) |
| 最初のpreview | 93.0s | 39.0s(import 1.4 + サンプリング 3.6 + bg_ref 0.6 + モデル読込 13.7 + 初回推論ウォームアップ。サーバーはモデルをキャッシュするので2件目以降はさらに短い) |
| 総時間 | 339s | 401s(負荷変動。本変更と無関係なpostprocessも91→131s) |
| 出力 | -- | **`results_dinosaur/Triceratops_matte.gif`とバイト一致**(旧2パスはS1 977,093で不一致、新はS1 975,023で基準と同一) |

旧2パスとの差は121フレーム中6フレーム(切り落とし検知フレームの区間修復 vs 単一フレーム全画面再推論)のみ。
`results_dinosaur/`は41b3591以前(9e3e6c1)の生成物なので、再生成は不要だった。

**1-4 ETAのdetect/birefnetをフレーム比例に**(commit 05f6155): YOLOX(640)とBiRefNet(1024²)は入力サイズ
固定なのでコストは解像度に比例しない。段名を`detect_frame`/`birefnet_frame`(フィールド`s_per_frame`)に
変え、`timing_stats.json`に旧名で残る過大なper-MPix EMAは読まない。種係数: cuda 0.12/0.31 s/frame
(上記Triceratops実測 detect_s 14.0s・birefnet_s 37.8s / 121f)、cpu 0.75/35 s/frame(旧cpu種は8〜17倍過小)。

### 第8回監査レバー3・4: 既定エンコーダを ss_alpha_gif へ、despill を band_only+est_scale=0.5 へ (adopted 2026-09-22)

第8回監査計画のレバー3(postprocess の無駄削減は既に2026-09-14に別コミットで採用済み)、
残るレバー2(flow_scale/preset)・3(despill est_scale)・4(ニューラル経路への ss_alpha_gif 適用)を
`tool/scripts/speed_experiment.py` で事前登録基準つきで実測した。設計は
V1〜V4(2026-09-02/03)の教訓を踏まえ、**despill/postprocess フェーズは軽量エンコーダ
(ss_alpha_gif)でレンダリングしてBiRefNet推論をクリップごとに1回だけ共有キャッシュし、
encoder 比較フェーズだけ実際の候補エンコーダ(ss4含む)を使う**という構成(過去の
V2/V3の「F2悪化」が実はBiRefNetの実行毎非決定性だったという2026-09-03の判明を踏まえた設計)。

**フィクスチャ**: 自然背景 `Triceratops.mp4`(F1i/F2/E2は評価不能、S1/E1のみ有効)+
フラットクロマ `sample/flatchroma2/` 4本(`--use-keyer off` でニューラル経路強制、
F1i/F2/E2が評価可能)。判定基準は「4本(自然背景含む)すべてで満たす」。

**despill(レバー3、C_band_only_half = band_only=True + est_scale=0.5)**: 4クリップ全てで
F1i/F2が現行(A_current)と**完全一致**(despillはalphaを触らないため当然)、E2_fringe_quality
は**改善**(比率1.05〜1.06、悪化なし基準0.97を大きく上回る好転)、despill時間は
31〜56s→18〜29s(約40〜47%減)。**採用**: `despill_band_only=True`,
`despill_est_scale=0.5`(旧: `False`/`1.0`)。B_band_only単独も基準を通過したが速度改善が
ごくわずか(5〜20%)なのでCのみ採用。

**postprocess(レバー2、flow_scale/flow_preset)**: `half_medium`(flow_scale=0.5)は
Triceratops単体ではS1 +5.0%以内・E1 -0.28%で通過したが、フラットクロマ4本では
**S1が+6.3%〜+9.1%で全て基準(≤+5%)を超過**(`full_fast`/`half_fast`も同様に不合格)。
**不採用**。флоの半解像度計算は光流の精度をこの素材では犠牲にしすぎる。既定
(`flow_scale=1.0, flow_preset="medium"`)を維持。

**encoder(レバー4、ss_alpha_gif/ss2/ss3をニューラル経路へ)**: フラットクロマ4本全てで
F1i +1.1%〜+3.3%(ss_alpha_gif)、E1 ±0.2%以内、S1 ±0.4%以内、F2はほぼ横ばい(むしろ
3/4クリップで改善)。**最悪フレームのマゼンタ合成による目視確認**(4クリップ×baseline/ss2/
ss_alpha_gif)でも新規の可視欠陥なし(口内の穴・フリンジパッチいずれも無し)。
encode時間は422〜556s→27〜45s(ss_alpha_gif、約15〜20倍)。**採用**: `encoder`既定を
`supersampled_gif`→`ss_alpha_gif`に変更。V2/V3(2026-09-02/03)がこの同じエンコーダを
「F2悪化」で不採用にしていたのは、後に判明した通りBiRefNetの実行毎非決定性が原因であり
(2026-09-03のV4節参照)、今回は共有キャッシュ推論でその交絡を排除して再検証し、正当性が
覆った。`supersampled_gif`はopt-inとして引き続き利用可能。

**反映**: `tool/pipeline/config.py`(despill_band_only/despill_est_scale/encoderの既定値、
docstring)、`tool/pipeline/__main__.py`(CLIの`--encoder`未指定時の実際のデフォルトが
`PipelineConfig`と食い違っていたバグも同時に修正)、`results_dinosaur/Triceratops_matte.gif`
を新既定で再生成(F3 +0.5%・S1 -0.6%・S2 -13%・E1改善、いずれもノイズ内で回帰なし、目視も
異常なし)、`server/eta.py`のdespill種係数(0.55→0.25、実測値で更新)。
`results_flatchroma2/`(keyer経路)は既にss_alpha_gifを使用しておりdespillも通らないため
無変更(既定値フリップの影響を受けない)。

回帰ゲート(`test_regression_gates.py`)18/18 pass + 1 skip(F1i on natural-bg Triceratops、
既知の仕様)。

### 第11計画 Part 3: 主被写体成分フィルタ(既定OFF・不採用)・人物箱の門・keyerの縁色unmix(既定ON)・t_lo実測フロア(既定OFF・不採用) (2026-09-23/24)

**3-2 採用**: `matte_core.subject_box`にクラス0(person)かつconf>=0.8の箱には
`min_box_frac`を適用しない例外を追加。widepose.mp4(灰色背景、彩度4.47のため
ニューラル経路)フレーム34-47は人物箱12.6-14.9%(conf 0.90-0.93)が
`min_box_frac=0.15`を割って棄却され、全画面BiRefNetがダンベル・ボトル・
ノートPCを前景化していた。実測: フレーム40の余分成分2(1140/1003px)->0、
クリップ全体で余分成分0.306->0.220個/frame、余分画素337.6->247.2px/frame。
cyclorama 4本(488フレーム全数)は該当フレーム0(影響なし、回帰なし)。

**3-1 実装したが不採用(既定OFF)**: `stages.keep_main_subject`(連結成分の
うち最大/最大の25%以上/その被写体のYOLOX箱に接するもの以外を透明化)を
実装したが、**事前登録ゲート不合格**。wideposeのフレーム110/180では開脚
ポーズで人物のYOLOX箱自体が床の小物(ノートPC・ボトル・ダンベル)まで
覆ってしまい、規則(iii)がそれらを「被写体に接する」として残す
(クリップ全体で除去は4/245フレーム・1339pxのみ、フレーム40/110/180の
目標である「余分成分->0」は40のみ達成)。閾値やルールを結果を見た後に
調整することは事前登録により禁止したため、**既定はOFF、opt-inとして
コードは残す**。副作用として: dinosaur/Triceratops.mp4(自然背景、ONに
すると2/121フレームで11px除去、F1i/S1はほぼ不変349025->975016 vs
975023)、cyclorama 4本ではON/OFFでF1i/S1/F2完全一致(該当フレームなし)。
今後、色や物体クラスでの絞り込み(例: YOLOXの他クラスや手に持つ物体の
検出)を追加すれば規則(iii)を狭められる可能性があるが、今回のスコープ外。

**3-3 採用(既定ON)**: `keyer.key_frame`に`keyer_unmix_coverage`を追加。
表示用α(ramp_k=0.5、現状維持)とは別に、t_hiをneutral_distまで広げた
「色の分離用被覆率」を縁帯域(display alpha>=0.996の画素も含む、2px膨張)
に適用してunpremultiplyする。表示αは無変更なのでF1i/S1は理論上不変 --
実測: purplebg 4本+flatchroma2 4本の計8クリップで**全てF1i/S1完全一致**
(unmix単独、keep_main_subject無効で分離計測)。背景色に近い外周画素
(outline_bgcol_px/frame): purplebg 381.8/678.6/487.1/310.5 -> 0.0/0.02/0.0/0.02、
flatchroma2 100.9/110.9/81.4/93.5 -> 0.8/0.7/0.5/0.5。E2(縁の色品質、
高いほど良い): purplebg 24.0/26.2/24.9/30.6 -> 36.3/36.9/37.2/46.2、
flatchroma2 52.6/49.4/54.6/53.2 -> 69.5/68.6/71.0/69.9。
`results_flatchroma2/`を新既定で再生成(F1i/S1は0.3%未満の差 -- 前回生成
[a753c9f, 2026-09-14]以降の無関係な変更による誤差範囲内、E2は全4クリップ
改善)、回帰ゲート18 passed + 1 skip(既知)、最悪フレーム(①感謝 frame 39)
目視で縁のフリンジ・欠損なし。

**3-4 実装したが不採用(既定OFF)**: `keyer_measured_t_lo`(t_loの下限を
クリップ自身の背景距離分布のp99.9から導出)も実装したが、**事前登録ゲート
「F1i不変」に不合格**。t_loが実際に動いたクリップだけF1i/S1が悪化する
明確な相関を確認(t_lo不変のクリップはF1i完全一致): purplebg 喜び
t_lo 3.0->4.0でF1i +28.1%(110124->141073)、感謝 3.0->3.3でF1i +9.7%、
挨拶 3.0->3.2でF1i +2.7%、思考は3.0のままF1i不変。webp遠景もや
(webp_soft_far_px/frame)は動いた分だけ大きく減る(喜び 32586->11038など)
が、F1iとのトレードオフが事前登録基準を超えるため不採用。既定OFF、
opt-inとして残す。

**keep_main_subjectをkeyer経路に適用(3-1のkeyer側)**: 島(motion-blur
による小さな離れ成分)は規則(i)(ii)のみ(箱なし)で除去可能 -- purplebg 4本
で島px/frame: 喜び105.8->0.0(max4348px)、思考20.4->0.0、感謝37.7->0.0、
挨拶39.4->0.0(max607px)。ただし副作用としてF1iが1.2-9.7%悪化する
(喜び+9.7%・思考+5.6%・感謝+6.9%・挨拶+1.2%) -- 離れ島がF1iの「内部
誤消去」計算に部分的に寄与しているため。3-1全体が既定OFFなので、この
keyer側適用も既定では効かない。

### YOLOX crop box slices the subject on the non-chroma route -- fixed and verified (found + implemented 2026-09-14, real-render verified 2026-09-16)

Rendering `sample/flatchroma2/①感謝.mp4` through the BiRefNet route
produced F1i=505,397 -- 2.5x over this repo's own permanent gate (<200,000 in
`tool/tests/test_regression_gates.py`). Magenta-composite inspection of the
worst frame (68) shows **the character's entire head and frill removed along a
straight horizontal line**. A straight-line cut is the signature of the crop
box, not of a semantic mistake: `Models.birefnet` mattes only inside
`subject_box`'s rect plus `margin=0.12`, and alpha is zero everywhere outside
it, so a box that under-covers the subject amputates whatever falls outside.
Diagnosed exactly on frame 68: YOLOX returned exactly one detection
(`cls=0 "person", conf=0.46, box y 330..642`, 16.5% of the frame -- comfortably
clears `min_box_frac=0.15`, which only rejects a box that's too SMALL in
*area*, not one that's normal-sized but mispositioned), while the true subject
(measured via the keyer's own colour-based extent on this flat-chroma frame)
spans y 114..643 -- the top ~180px of the head/frill sat entirely outside the
margin-padded crop.

**Fix implemented** (`tool/matte_core.py`): `Models.birefnet` now detects this
at the source. Its crop/letterbox/paste logic was extracted into
`_birefnet_crop_pass` (no behaviour change, pure refactor) so it can be called
twice. After the box-based pass, `_alpha_touches_a_crop_edge` checks whether
any confidently-opaque pixel (`>0.5`) sits on an edge of the crop rect that
ISN'T also the frame boundary (touching the true frame edge is normal -- the
subject legitimately leaves frame there; only an edge that came from the
box+margin, not from clamping to 0/W/H, indicates a cut). If so, ONE full-frame
(`box=None`) fallback pass replaces the result -- bounded to at most 2x cost,
never recursive (a full-frame pass's crop rect IS the whole frame, so none of
its edges can ever trigger the same check again). `Models.box_reinfer_frames`
counts how often this fired; `runner.infer_clip` reports it as
`timings["box_reinfer_frames"]` and a `[box]` log line.

Deliberately did NOT implement the plan's original `_salient_box`-plus-union
sketch: `_salient_box` operating on the crop-based `out` can only ever return
a box AT MOST as large as the crop itself (the array is zero outside it by
construction), so it cannot discover how far the true subject extends beyond
a crop that already cut it off -- using it there would have been decorative,
not functional. The full-frame fallback is simpler, strictly correct
regardless of how badly the original box undershot, and still bounded to one
extra call. `_salient_box` remains available (still used by the SAM2 route)
but isn't part of this fix.

Also fixed in the same pass: `keyframe_alpha` (the image-pipeline/SAM2 route)
now defaults `min_box_frac=0.15` on its own `subject_box` call -- it had none
(defaulting to `subject_box`'s own permissive `0.0`), contradicting
`config.py`'s B2 note that "every OTHER caller" already had this gate. That
function currently has zero call sites anywhere in this repo, so the change
is behaviour-neutral today, but a future caller inherits the same protection
`tool/pipeline/runner.py` already has.

**Verified, both control flow and real render.** Control flow:
edge-detection on all 4 crop edges, frame-boundary exemption on all 4, the
fallback firing exactly once and never recursing, `box=None` never
re-triggering itself -- via `tool/tests/test_birefnet_truncation.py` (8
tests, a fake stand-in for `Models` so no real ONNX/GPU is involved).

Real render (2026-09-16, GPU freed up): `①感謝.mp4` through the neural route
(`--use-keyer off`) --

| | before this fix | after |
|---|---|---|
| F1i (primary gate, <200,000) | 505,397 | **20,635** (24.5x) |
| F2 | -- | 319 |
| S1 | -- | 99,169 |
| S3 | -- | 0 |

**39 of 124 frames (31%) triggered the full-frame fallback** -- the crop was
under-covering the subject on nearly a third of the clip, not just the one
frame (68) the original diagnosis singled out. Frame 68 itself, and every
other sampled frame (0/42/68/100), confirmed by magenta-composite: the head
and frill are intact where they used to be sheared off along a straight
line. Encode/inference cost is unaffected by the fix itself (the fallback
only adds one extra BiRefNet call on the ~31% of frames that need it; total
wall time for this clip was 687s, dominated as always by the `supersampled_gif`
encode -- see the keyer entry above for why that's irrelevant to a clip this
small once the keyer path is available, but this clip's subject/backdrop
combination doesn't gate as flat-chroma-enough on its own colour design, so
`--use-keyer off` here is purely to force the comparison, not how this clip
would actually be served).

**`cyclorama` real-footage regression check** (4 clips, `喜び/思考/感謝/挨拶`,
1572x1316, `device=cuda`, no `--use-keyer` flag needed -- a white cyclorama
never gates as keyable, see the keyer entry above) against `results_cyclorama/`,
the pre-existing accepted baseline:

| clip | F1i (new) | F1i (base) | F2 (new) | F2 (base) | S1 (new) | S1 (base) | box_reinfer |
|---|---|---|---|---|---|---|---|
| 喜び | 20,409 | 40,591 | 1,111,362 | 1,153,181 | 227,111 | 289,600 | 8/122 (6.6%) |
| 思考 | 6,634 | 25,825 | 1,110,285 | 1,145,634 | 174,555 | 246,836 | 8/122 (6.6%) |
| 感謝 | 1,733 | 77,364 | 975,568 | 1,014,475 | 132,364 | 233,736 | 17/122 (13.9%) |
| 挨拶 | 9,931 | 25,322 | 993,178 | 993,837 | 173,695 | 213,121 | 19/122 (15.6%) |

**Every metric improved on every clip -- not merely "no regression".** F1i
dropped 2.5x-45x, F2 dropped slightly on 3/4 (flat on the 4th), S1 (chatter)
dropped 10-44%. The fallback fired non-trivially on this set too (6.6-15.6%
of frames per clip) -- real footage's YOLOX "person" detection undershoots
the actual extent (hair, raised arms, loose clothing) more often than the
mascot batch's cleaner silhouette. Since the fallback only ever REPLACES a
detected-truncated crop with a full-frame pass, it structurally cannot make
a clip worse than before the fix -- these numbers are the confirmation, not
a surprise. Visual check (magenta composite, one mid-clip frame per clip):
clean full-body extraction on all four, no fringe/amputation artifacts.

**Conclusion: adopted, no follow-up needed.** The fix requires no config
flag and is unconditional in `Models.birefnet` -- every non-keyer clip
already benefits from it with no further action.


### Colour-only keyer as the flat-chroma fast path (2026-09-14)

`tool/pipeline/keyer.py`, auto-selected by `PipelineConfig.use_keyer="auto"`.
Measured on `sample/flatchroma2/①感謝.mp4` (768x768, 124 frames),
both runs on the same loaded machine:

| stage | BiRefNet route | keyer |
|---|---|---|
| YOLOX detect | 11.4s | 0 |
| BiRefNet | 34.6s | 0 |
| despill | 25.4s (pymatting ML, whole crop) | included below |
| mc_median (optical flow) | 57.1s | 0 |
| keyer (distance + ramp + unpremultiply) | — | 6.6s |
| encode | 604.1s (`supersampled_gif` ss=4) | 18.4s (`ss_alpha_gif`) |
| **total** | **737s** | **29s** |

All four clips land at 29-32s. Quality on ①感謝 (harness, same source):

| metric | baseline | keyer |
|---|---|---|
| F1i false-erase-interior (primary gate) | 505,397 | 11,833 |
| F2 false-keep | 275 | 0 |
| S1 chatter | 326,659 | 149,488 |
| S2 colour flicker | 918 | 687 |

**Why the expensive stages disappear rather than get optimised.** Each one
exists to compensate for a BiRefNet weakness that a flat backdrop doesn't
produce: the detector only exists to give BiRefNet a crop box; mc_median only
exists to damp its frame-to-frame chatter; the ML despill only exists because
the background colour is unknown; and the two-pass 4x encoder only exists to
hide its run-to-run alpha non-determinism near the 1-bit threshold (the V4
root-cause finding below). A keyed matte is a deterministic per-pixel function
of the source with a *known* background colour, so all four justifications
lapse at once. Notably S1 came out 2.2x BETTER than the neural route *with*
57s of optical-flow smoothing, which is the cleanest confirmation that the
chatter was BiRefNet's, not the content's.

**Gating.** Two conditions, both measured from the backdrop alone, both
required (`keyer.is_safe`): the backdrop is FLAT, and its colour is
SATURATED enough to key on (>=20 Cb/Cr levels from neutral).

The second condition is not redundant, and finding that out mattered. The
first version gated on flatness plus a "subject is far from the key colour"
margin, and it engaged on the **cyclorama real-footage set** -- which is a white
studio cyclorama, perfectly flat (frac_bg_like=1.0000) and completely
unkeyable, since a neutral backdrop cannot be told apart from the subject's
own white/grey areas. Measured backdrop saturation separates the sets by 4x
with no overlap:

| set | saturation | keyable |
|---|---|---|
| flatchroma2 (blue `#394ABF`) | 61.0 | yes |
| purplebg (purple) | 42.1 | yes |
| dinosaur/Triceratops (forest) | 10.0 | no |
| cyclorama (white cyclorama) | 2.0 | no |
| movia (office photo) | 1.4 | no |

`use_keyer="on"` cannot bypass either check.

**A vacuous self-check, caught by its own output.** The dropped "subject
margin" test was circular: it located the subject by thresholding chroma
distance at 60 sigma and then reported the minimum distance *within that
region*, which is >=60 by construction. Every clip therefore reported exactly
"subject 60 sigma clear" and the margin always passed -- the gate did nothing
while reading as protective, and it pinned the derived ramp to a constant too.
The tell was the identical number on every clip. This is the plan's structural
problem #2 ("処理が動いていないことに気づけない") reproducing itself inside a
check written specifically to avoid problem #1; worth remembering that a
self-check needs its own evidence that it can ever fail.

The residual risk it pretended to cover -- a subject genuinely containing the
key colour -- is not decidable from colour alone (Smith & Blinn 1996). It is
handled where it actually can be: at render time, where the key colour is
chosen explicitly (this batch records it in `report.md`).

**Ramp derivation.** The QC metrics' own `T_LO`/`T_HI` (3/8 sigma) are
calibrated for *classifying* pixels and are unusable for *generating* alpha:
with sigma pinned at its 1.0 floor on a clean render and the subject ~100
sigma away, T_HI=8 makes a pixel only 8% of the way from backdrop to subject
fully opaque, stranding the boundary's blended pixels at alpha=1 with their
key colour intact. The ramp top is therefore derived per clip as
`keyer_ramp_k` x (measured distance to the nearest confident-subject pixel).

### E1/E2 are not valid cross-matte comparators (2026-09-14)

Found while grading the keyer. `E1_perimeter_ratio` and `E2_fringe_quality`
were introduced (Phase 0 / V4) to compare an ENCODER or DESPILL change on the
*same* matte, where the silhouette is fixed and only edge rendering or colour
changes. They are not valid for comparing two mattes whose silhouettes sit in
different places, because both reward cutting INSIDE the subject.

Demonstrated rather than argued -- eroding the keyer's own matte, which
strictly destroys real subject content, monotonically *improves* both:

| erode | F1i (lower better) | E1 (lower better) | E2 (higher better) |
|---|---|---|---|
| 0px | 11,908 | 7.61 | 51.0 |
| 1px | 22,496 | 6.91 | 70.7 |
| 2px | 36,763 | 7.25 | 87.2 |
| 3px | 55,165 | 6.88 | 93.8 |
| *baseline* | *505,397* | *6.71* | *86.6* |

The baseline's better E1/E2 are fully explained by it cutting ~2px inside the
true silhouette everywhere (its F1i is 42x worse). Treat both as within-matte
diagnostics only; cross-matte decisions rest on F1i/F2/S1/S2 plus visual
inspection.

### Continuous (not binary) temporal hole-fill

`_topology_temporal`'s hole-fill (video person route, GPU only) used to patch
a hole with a flat 1.0 whenever the flow-warped previous frame was foreground
there. It now fills with the warped previous frame's *actual* alpha value
instead, preserving soft/semi-transparent structure a hard fill would
flatten. Verified on `sample/widepose.mp4` (first 100 frames, full GPU
pipeline): GT IoU 0.958->0.959, Bnd-F 0.992->0.993, worst-frame IoU
0.924->0.947 (the frame that benefits most from a hole-fill is exactly where
this helps), `dev/qc_matte.py` mean flicker 0.05%->0.04%. A small,
consistent, no-downside win, kept.

### despill (foreground color estimation)

Every output pixel used to be the raw photographed RGB value, even at
semi-transparent edges (hair, fur, motion blur) — those pixels still carry a
`(1-alpha)` contribution from the *original* background color (`I = αF +
(1-α)B`), which shows up as a color fringe once recomposited onto a NEW
background. `matte_core.despill()` now solves for the true foreground color
`F` via `pymatting.estimate_foreground_ml` (Germer et al. 2020's multi-level
foreground estimation — already an MIT dependency here, no new package;
`rembg`/`backgroundremover` use the same function, just gated behind an
opt-in flag most users never enable) before compositing, cropped to the
subject's bbox+padding the same way `_matte_refine` is. Measured overhead on
this project's dev CPU: ~0.08s on top of BiRefNet's ~10.5s/frame (<1%) —
alpha itself is unchanged, so GT scores are unaffected; the fix is invisible
on this repo's two GT clips (both fairly neutral-colored backgrounds) but
real on a colored backdrop. Applied in both `bg_remove_image.py` (skipped
under `--fast`) and every alpha-producing branch of `bg_remove_video.py`.

### Input-size / disk-space / SAM2-timeout hardening

Cheap pre-release insurance for a CLI an end user runs on their own machine
(not a server under adversarial load): `matte_core.check_frame_size()`
rejects (with a clear message) any frame over 50 megapixels unless
`--allow-large` is passed, on both CLIs; `check_disk_space()` estimates video
output size and aborts before a multi-minute encode if free space looks
insufficient; `_Sam2Worker.mask()` now times out after 60s (well above any
observed real call) instead of hanging forever if the SAM2 subprocess
crashes or wedges. All three are narrow, cheap gates, not a full
resource-governance system.

## Researched but not adopted

### keep_main_subject規則(iii)のSAM2画素マスク化(第12計画) — 事前登録ゲート不合格、既定OFF維持(2026-09-24)

第11計画Part 3-1で既定OFFのまま出荷された`keep_main_subject`の規則(iii)
(「YOLOX人物箱の矩形内に画素が1つでもある成分は残す」)を、SAM2単画像予測
(`matte_core._Sam2Worker`、Apache-2.0、既存基盤)による人物画素マスクで
置換する試み。矩形ベースの規則(iii)は、widepose.mp4の開脚ポーズ
(frame 110/180)でボトルが人物箱に完全内包される、原理的に分離不可能な
ケースで失敗していた(第11計画の記録参照)。

**実装**: `tool/pipeline/stages._component_keep_labels`に`sam2_mask`引数を
追加。規則(iii')「膨張(dilate_px=5)したSAM2人物マスクとの重なり率>=0.30
(`keep_main_subject_sam2_overlap`)の成分は残す」に置換。SAM2マスクの
健全性チェック(`keep_main_subject_sam2_min_mask_frac=0.20 <= mask_area/box_area
<= 2.0`)に落ちた場合は旧・矩形ルールへフォールバック。`runner.run_clip`が
クリップ単位で軽量な`_Sam2Worker`を生成し(GPU限定、keyer経路は対象外)、
規則(i)(ii)で結果が決まらないフレームだけ呼ぶ(コスト最小化)。
閾値は全て事前登録(0.30/5px/0.20、DECISIONS.md記載前に固定、結果を見て
動かしていない)。

**事前登録ゲート**: (1) widepose frames 40/110/180で余分成分0、
(2) dinosaur+flatchroma2 4本でF1i/S1がノイズ床内、(3) postprocess_s増分
<=60s/クリップ、(4) SAM2フォールバック率<=10%。`tool/scripts/
sam2_keep_main_subject_experiment.py`で実測(GPU、flatchroma2 4本を
`use_keyer="off"`でニューラル強制)。

**実測結果**:

| clip | F1i(baseline→lever) | S1(baseline→lever) | postprocess増分 | SAM2フォールバック |
|---|---|---|---|---|
| widepose | 212377→213125(実質不変) | 685.4→689.6(実質不変) | +34.5s | **11/23フレーム=47.8%** |
| dinosaur_Triceratops | null(自然背景) | 8125.2→8123.5(実質不変) | (GPU競合でノイズ支配、per-stage timingは+29.7s) | 0/16=0% |
| flatchroma2_c1 | 21306→21306(完全一致) | 803.0→803.0(完全一致) | +28.6s | 0/13=0% |
| flatchroma2_c2 | 116117→116117(完全一致) | 2505.27→2505.29(誤差) | +29.2s | 0/15=0% |
| flatchroma2_c3 | 17364→17364(完全一致) | 779.49→779.49(完全一致) | +29.6s | 0/16=0% |
| flatchroma2_c4 | 24682→24682(完全一致) | 845.42→845.42(完全一致) | +28.5s | 0/7=0% |

ゲート(2)(3)は全クリップで合格(dinosaur+flatchroma2でF1i/S1が実質不変、
postprocess増分も上限内)。**ゲート(1)(4)は主目標のwideposeで不合格**:

- **widepose frame 110**: baseline `[28332, 2863, 1147, 1117, 37, 9]`
  → lever `[28332, 2863, 1147, 1117]`(37px/9pxの微小ノイズのみ除去、
  **ノートPC(2863)・ボトル(1147)・ダンベル(1117)は全て残存**)
- **widepose frame 180**: baseline `[27561, 2542, 1140, 989]` →
  lever `[27561, 2542, 1140, 989]`(**完全に無変化**)
- **wideposeクリップ全体のSAM2フォールバック率47.8%**(11/23フレーム、
  基準10%を大幅超過)

**根本原因(SAM2単独の追加診断で特定、`_Sam2Worker.mask`をクリップの
実フレーム/実箱で直接呼び出し確認済み)**: SAM2のマスク自体は正しく
機能していた。frame 110のSAM2マスク面積(30,768px)は人物の連結成分
面積(28,345px)にほぼ一致し、小道具を正しく除外していた:

| frame | 箱面積 | SAM2マスク面積 | frac(マスク/箱) | 健全性チェック(0.20-2.0) |
|---|---|---|---|---|
| 40(通常姿勢) | 66,783 | 26,279 | 0.393 | 合格 |
| 110(開脚) | 155,610 | 30,768 | **0.198** | **不合格(0.20をわずかに下回る)** |
| 180(開脚) | 156,618 | 30,092 | **0.192** | **不合格** |

問題は健全性チェックの`min_mask_frac=0.20`という**固定床が、開脚姿勢
という「まさに直したかったケース」を体系的に誤判定する**ことにあった:
手足が箱を大きく広げる一方、人物本体のシルエットは箱面積に対して
構造的に小さくなる(これはSAM2の失敗ではなく、ワイドポーズの幾何学的
必然)。frac=0.198/0.192は0.20という事前登録した床をわずかに下回った
だけで、SAM2は実際には正しい判断をしていたのに、健全性チェックが
それを「SAM2が被写体を見失った」と誤認し旧・矩形ルールへフォールバック
させていた。

**結論**: ablation原則(事前登録した閾値は結果を見て動かさない)に従い、
この場で`min_mask_frac`を再調整してゲートを通すことはしない。
`keep_main_subject`/`keep_main_subject_sam2`は既定値(共にコード上は
`keep_main_subject_sam2=True`だが`keep_main_subject`自体が既定False
のため実質無効)のまま据え置く。SAM2画素マスクという設計方針自体
(規則(iii)をアルファ非破壊の成分判定にのみ使う)は妥当性が確認された
(dinosaur/flatchroma2でF1i/S1が完全一致、副作用ゼロ)。フォールバック時の
振る舞いも安全(旧ルールへの後退のみ、新規の劣化なし)。

**将来の再訪候補(未検証、次にこの機能を再訪する場合の出発点として記録、
このセッションでは実装・検証しない)**:
1. 健全性チェックの正規化基準を「箱面積」ではなく「規則(i)(ii)で既に
   keepと決まった最大成分自身の面積」に変更する(ワイドポーズでも
   人物本体の面積は箱の形に依存しないため、より頑健な基準になる可能性)
2. 健全性チェックを撤廃し、SAM2マスクが空でない限り常に採用する
   (フォールバックという「安全側」の機構自体が今回悪さをした)
3. `dilate_px`を大きくし、SAM2マスクの穏やかな過小評価を吸収する

いずれも新しい閾値であり、採用するなら**それ自体を事前登録した独立の
ゲートで再検証**すること(このゲートの事後修正としては扱わない)。

`data/keep_main_subject_sam2_experiment/`(109MB、baseline/lever計12本の
GIF+config+results.json)はこの記録に転記済みのため削除可(ディスク衛生)。


### box_mode (レバーC: クロップ箱の時間安定化) — clip_union/smoothed both measured, both rejected (2026-09-22)

第9計画6.Cの実装: `tool/pipeline/runner.py`のYOLOX検出(`Models.subject_box`)を
2パス化した(パス1で全フレームYOLOX、パス2でBiRefNet)。箱の決め方に
`PipelineConfig.box_mode`("per_frame"|"clip_union"|"smoothed")を追加し、
`tool/pipeline/boxmode.py`に実装: `clip_union`は全フレーム箱のunion一本、
`smoothed`は窓15の中央値フィルタ+サイズヒステリシス(拡大即時、縮小はN=5
フレーム継続)。切り落とし検知のフォールバックも単一フレームの全画面ジャンプ
(`matte_core.Models.birefnet`の`auto_full_frame_fallback`)から、クリップ単位
(`boxmode.repair_truncated_regions`: 切り落とされたフレームを連続区間にまとめ、
区間の箱をunion+50%成長させて再推論)に変更した。既定は`box_mode="per_frame"`
のまま(挙動不変)。

**事前登録基準**: flatchroma2 4クリップ(`①感謝/②喜び/③思考/④挨拶`, `use_keyer="off"`
でニューラル経路強制)+ Triceratops(自然背景、常にニューラル)の計5クリップ全てで
同時に (1) S1改善(ノイズ床超) かつ (2) F1i ±2%以内 かつ (3) `box_reinfer_frames`==0。
`tool/scripts/box_mode_experiment.py`を新設し測定(`per_frame`を2回実行して
ノイズ床を確認 -- 結果は全指標で完全ビット一致だったため、このマシン/設定での
ノイズ床は実質0。第8回監査で確認済みの「HEURISTIC cudnn algo searchはこの
RTX5090上で決定的」という知見と整合)。

**実測結果**(pass1 detect時間も記録。単位: S1/F1i/F2はpx、box_reinferはフレーム数):

| clip | variant | S1 (Δ) | F1i (Δ) | box_reinfer_frames | detect_s |
|---|---|---|---|---|---|
| ①感謝 | per_frame(基準) | 98905 | 21321 | 49 | 11.4/6.3 |
| | clip_union | 95125 (-3.8%) | 22992 (**+7.8%**) | **0** | 7.2 |
| | smoothed | 96723 (-2.2%) | 21005 (-1.5%) | **69**(基準より悪化) | 5.6 |
| ②喜び | per_frame(基準) | 308194 | 116062 | 3 | 6.4/6.0 |
| | clip_union | 303327 (-1.6%) | 113175 (**-2.5%**) | **0** | 5.6 |
| | smoothed | 306612 (-0.5%) | 116845 (+0.7%) | **64**(基準より悪化) | 5.6 |
| ③思考 | per_frame(基準) | 95199 | 17309 | 17 | 5.7/5.6 |
| | clip_union | 89705 (-5.8%) | 18782 (**+8.5%**) | **0** | 5.6 |
| | smoothed | 93267 (-2.0%) | 17118 (-1.1%) | **3** | 5.5 |
| ④挨拶 | per_frame(基準) | 103983 | 24701 | 5 | 5.5/5.5 |
| | clip_union | 101123 (-2.8%) | 24615 (-0.4%) | **0** | 5.6 |
| | smoothed | 102906 (-1.0%) | 25135 (+1.8%) | 6 | 5.6 |
| Triceratops | per_frame(基準) | 977093 | n/a(非chroma) | 11 | 9.4/9.4 |
| | clip_union | 950081 (-2.8%) | n/a | **0** | 9.5 |
| | smoothed | 932062 (-4.6%) | n/a | **0** | 9.6 |

`per_frame`を2回実行(A/B)した結果は5クリップ全てでS1/F1i/F2/E1_medianが
完全ビット一致 -- このマシンではBiRefNetのフレーム推論が決定的であることを
再確認できた(ノイズ床=0)。

**判定(基準どおり、閾値は緩めていない)**:
- **S1**: 両変種とも5クリップ全てで改善(ノイズ床が0のため、この改善は測定誤差
  ではなく実際の効果)。この基準単体は両方パス。
- **F1i ±2%**: `clip_union`はflatchroma2の4クリップ中3クリップで超過(+7.8%/-2.5%/
  +8.5%)、④挨拶のみパス。`smoothed`は4クリップ全てパス(最大|Δ|1.8%)。
- **`box_reinfer_frames`==0**: `clip_union`は5クリップ全てで0(構造上当然 --
  クリップ全体で単一の箱を使うため、どのフレームの生検出箱よりも小さくなり
  得ない)。`smoothed`は5クリップ中4クリップで非0、しかも①感謝(69、基準の
  49より悪化)と②喜び(64、基準の3より大幅悪化)では**是正しようとしている
  問題そのものを悪化させた** -- 中央値+ヒステリシスが速い被写体移動に
  追従し切れず、生の毎フレーム検出より遅れて箱が外れるケースがあると判明。

**どちらの変種も3条件を同時に満たすクリップは無い → 不採用**。`clip_union`は
安定性(S1・box_reinfer)は非常に強いがF1i精度を崩す(箱が被写体移動範囲全体を
覆うため実効解像度が落ち、BiRefNetの1024^2入力に対する被写体占有率が下がる
-- 計画が事前に指摘していたトレードオフどおり)。`smoothed`はF1i精度は保つが
安定性そのものの基準(box_reinfer_frames)を安定させられず、一部クリップで
悪化させる矛盾した結果になった。

**cyclorama目視・数値チェック**(`sample/cyclorama/` 4クリップ、非chroma・被写体
移動大、`results_cyclorama/`は一切変更せず別ディレクトリに出力): 12run(4クリップ×
3variant)全て完走、クラッシュなし。F1i/F2/S1/E1_medianは全variantでbaseline
と同じ桁・同じ傾向(S1はここでも両variantでbaseline比微改善、F1i/F2は
±20%以内の変動で異常なスパイクなし)。解像度崩壊や輪郭劣化を示す数値的な
兆候はなし。**注記**: 主基準(flatchroma2+Triceratops)が既に両変種を不採用と
判定したため、cycloramaはマゼンタ合成による詳細目視は実施していない(採否を
左右しないため) -- 数値チェックのみで異常なしを確認した。

**結論: 不採用、`box_mode`の既定は`"per_frame"`のまま変更なし**。
`tool/pipeline/boxmode.py`のコード自体(2パス化・3variantの解決ロジック・
クリップ単位の切り落とし修復)はオプトインとして残す -- Bレバー(SAM2由来の
箱)が採用されればそちらに置き換わる可能性があるほか、`smoothed`の
ヒステリシスパラメータ(window/shrink_hold)を調整すれば box_reinfer の
悪化が解消するかもしれず、再検討の土台として保持する価値がある。

**再検討する場合のヒント**: `clip_union`はF1iを悪化させる根本原因(箱が
大きすぎて実効解像度が落ちる)を解消しない限り採用不可 -- `_salient_box`
(SAM2由来、Bレバー)のようなタイトな箱と組み合わせるなら再検討の余地あり。
`smoothed`はshrink_hold(現在N=5)を増やす/減らす、またはwindowを短くする
ことで「追従の遅れ」対「ジッタ平滑化」のトレードオフを再探索できる余地が
残っている。

### Luma term in the keyer's distance (2026-09-14) — measured ineffective, deleted

The keyer's first prototype scored ~9% worse than the neural route on E1
(edge jaggedness). Hypothesis: these sources are h264 **yuv420p**, so Cb/Cr
are subsampled 2x2 and a chroma-only distance quantises the silhouette to 2px
blocks, while luma is always full resolution -- so adding a luma-difference
term should restore per-pixel edge detail.

Implemented and swept (weights 0.05/0.1/0.2 x ramp_k 0.3/0.4/0.5/0.65). The
hypothesis did not hold: E1 moved by less than 0.02 at every setting, and E2
got slightly *worse* with more luma weight. Deleted rather than shipped as a
dead knob (Phase 3 ablation discipline). The E1 gap was separately shown to be
a metric artefact, not a real defect -- see "E1/E2 are not valid cross-matte
comparators" above.

Widening the ramp (`keyer_ramp_k`) is the lever that does work; default 0.5,
which leaves a 2.0x safety margin on this fixture set against `is_safe`'s
required 1.5x.

### despill_band_only / despill_est_scale / flow_scale / flow_preset / faster GIF encoders (V6/V7, 2026-09-13) — inconclusive, deferred

Speed-lever fields (`despill_band_only`, `despill_est_scale`, `flow_scale`,
`flow_preset`) and encoder alternatives (`ss=2`/`ss=3`/`ss_alpha_gif` instead
of the default `supersampled_gif` ss=4) were added to `PipelineConfig` with
behavior-preserving defaults, and `tool/scripts/speed_experiment.py` was
built to measure them against pre-registered gates on the 4
`sample/dinosaur/{思考,喜び,感謝,挨拶}.mp4` clips (all 4 must pass for adoption
-- see the plan's 第6計画 Part C).

**Not adopted, deferred pending a full 4-clip run**: encode
(`supersampled_gif` ss=4) measured 1080-1513s/variant on this shared
machine during the one attempted run -- 2-7x slower than this project's own
historical measurements (200-230s uncontended, 420-600s under prior
contention). At that pace a single clip's full 11-variant sweep (3 despill
+ 4 postprocess + 4 encoder, most requiring a full encode) projected to
~4 hours; the run was stopped after 3 despill variants completed rather
than let one clip consume the whole session, with 4 clips clearly
infeasible in one sitting at this pace.

**Partial data (挨拶 clip only, informational -- NOT sufficient to adopt anything on its own)**:

| variant | despill time | F1i | F2 | E2_mean |
|---|---|---|---|---|
| A_current (baseline) | 136.6s | 109972 | 8752 | 76.907 |
| B_band_only | 136.0s | 109972 (identical) | 8752 (identical) | 75.713 (-1.55%) |
| C_band_only + est_scale=0.5 | 77.7s | 109972 (identical) | 8752 (identical) | 79.896 (+3.9%, better than baseline) |

Both B and C pass the pre-registered despill gate (E2_mean >= 0.97xA;
F1i/F2 byte-identical, as expected since despill never touches alpha) **on
this one clip**, but the pre-registered bar requires all 4. B's time
savings are negligible (136.6->136.0s, matching the plan's own prediction
that band_only alone wouldn't help much since pymatting's ML solve still
runs over the whole crop); C's ~43% reduction (136.6->77.7s) is close to
the plan's estimated 2x. postprocess/encoder variants never got a
data point on any clip.

**Resume when the shared machine is lightly loaded** (`uptime` load average
in the single digits, no other user's process pinning multiple cores --
`ps aux --sort=-%cpu`): `export LD_LIBRARY_PATH=...` per `run.sh` (the
CUDA execution provider silently falls back to CPU without it -- this
actually happened on the FIRST attempt of this run, caught by the
provider-verification warning added in an earlier audit), then
`venv/bin/python -m tool.scripts.speed_experiment --clear-cache --out data/speed_experiment/<date>`
for all 4 clips. The adoption gates (`judge()` in that script) are already
implemented and don't need to change.

**Follow-up investigation (2026-09-13, same day, real production incident):**
this same encoder (`supersampled_gif` ss=4) hard-failed a real user job with
`E_ENCODE_TIMEOUT` at 3605s under load average 60-97 on this 24-core
machine (16x its usual ~220s -- every OTHER pipeline stage only slowed
2-3x under the identical conditions). This triggered two responses: (1) a
new, shipped, load-aware fallback -- see "Load-aware encoder fallback"
below -- and (2) a check on whether `matte_core.py`'s `cudnn_conv_algo_
search=HEURISTIC` setting (the suspected root cause of the earlier F2
nondeterminism that blocked re-testing ss=2/3/ss_alpha_gif) still needs to
be HEURISTIC on this machine's actual GPU. Finding: the code comment
justifying it referenced a Turing/cc7.5 GPU, but this deployment's actual
GPU is a Blackwell-class one (cc12.0) -- EXHAUSTIVE, previously assumed to
crash here, in fact creates a session fine on this GPU. Re-tested
determinism directly: ran the real 挨拶.mp4 clip's inference through
`infer_clip` twice under HEURISTIC and once under EXHAUSTIVE, all as
separate process launches on an idle GPU -- all three produced
BYTE-IDENTICAL alpha (sha256 match across all 122 frames). No
nondeterminism reproduced under today's (uncontended-GPU) conditions with
either setting, and switching to EXHAUSTIVE is NOT clearly better in
theory either (HEURISTIC is a static shape-keyed lookup, unaffected by
concurrent load in principle; EXHAUSTIVE times candidate kernels at
runtime and could in principle pick a different winner under different
contention -- no basis to expect it to be MORE stable under load).
**Left on HEURISTIC** (the comment in `matte_core.py` was corrected to
stop citing a GPU this deployment doesn't have, but the setting itself was
not changed on inconclusive evidence). The original F2 defect (2026-09-03)
was never reproduced under deliberately-created GPU contention (this
investigation didn't attempt that), so it remains an open, unresolved
question whether that defect was contention-dependent nondeterminism or
something else entirely. Practically, this does NOT block re-running
`speed_experiment.py`: its own design (one shared cached inference reused
across every despill/postprocess/encoder variant per clip) already
isolates the comparison from BiRefNet's own run-to-run behavior,
whatever its cause.

### Load-aware encoder fallback (shipped, 2026-09-13)

Rather than wait for a lightly-loaded window to re-validate a faster
default encoder (still pending, see above), shipped a narrower, immediately
safe mitigation for the actual observed failure (E_ENCODE_TIMEOUT under
heavy contention): `server/jobs.py`'s `LOAD_AWARE_ENCODE` (default on)
checks `server/probe.cpu_load_ratio()` (1-minute load average / core
count) at the moment each clip starts, and substitutes `webp` for the
default `supersampled_gif` if the ratio exceeds `LOAD_THRESHOLD` (default
1.5) -- never for an explicitly-requested encoder. `webp` (not
`ss_alpha_gif`) was chosen deliberately: it carries true 8-bit alpha (no
1-bit threshold snap), so it's structurally immune to the exact defect
class (`ss_alpha_gif`'s past visible F2 patch, alpha flipping across the
128 threshold) that's still an open question above -- adopting it as a
*fallback* doesn't require first resolving that question, unlike adopting
`ss_alpha_gif` or a lower supersample factor as the new *default* would.

### B-1: SAM2 動画追跡 VRAM 実測 -- 予算内、下記スタブの VRAM 懸念を実測で更新 (2026-09-22)

第9計画 Part B のステップ1(`tool/scripts/sam2_seed_experiment.py --measure-only`)。
直下の「SAM2 video-tracking mode」スタブが不採用理由の一つとして挙げていた「VRAM が
クリップ長に応じて増大し続ける(GitHub issue: 2分30秒/24fps で~60GB)」という懸念を、
実測で直接検証した。**結論: この GPU(32GB)はもちろん、既存の 8GB cold-start 予算
(`HEROEXTRACTOR_MIN_FREE_MB`、`server/jobs.py:156` の既定値)にも余裕で収まる** --
Lever B は生きており、次段(B-2 プロトタイプ)に進んでよい。ただし本コミットは
**測定のみ**(B-2/B-3 のトリマップ/シード機能は未実装、別コミット)。

**方法**: `sample/dinosaur/Triceratops.mp4`(1656x1248, 24fps, 121フレーム=約5秒しか
無いため 300/900/1800 目標フレーム数に対してフレームをループして 10/30/60秒(240/720/
1440フレーム、24fps基準)を作成)を 1024x1024(SAM2既定解像度)に INTER_AREA でリサイズ
し JPEG 連番へ書き出し(`init_state` はパス入力のみで decord 未導入のため mp4 直読み不可、
既知の制約)。`_Sam2Worker`(`matte_core.py:504-585`)と同型の spawn 子プロセスで
`build_sam2_video_predictor` → `init_state(offload_video_to_cpu=True,
offload_state_to_cpu=True, async_loading_frames=True)` → フレーム0に中央プレース
ホルダー box(精度は測定に無関係)→ `propagate_in_video()` を実行し
`torch.cuda.max_memory_allocated/reserved` を記録。親プロセスから
`nvidia-smi --query-compute-apps` を0.5秒間隔でポーリングしクロスチェック。各長さで
「SAM2 単体」と「同居」(親プロセスが `runner._load_models` で BiRefNet+YOLOX を実際に
GPU へロードした状態で SAM2 子プロセスを実行、本番の `track` ステージ相当)の両方を測定。

**実測値**(全て OK、クラッシュ/OOM 無し):

| 長さ | フレーム数 | シナリオ | torch max_allocated | torch max_reserved | nvidia-smi 実測ピーク |
|---|---|---|---|---|---|
| 10s | 240 | 単体 | 774MB | 1149MB | 1726MB |
| 10s | 240 | 同居 | 774MB | 1149MB | 2698MB |
| 30s | 720 | 単体 | 775MB | 1149MB | 1726MB |
| 30s | 720 | 同居 | 775MB | 1149MB | 2698MB |
| 60s | 1440 | 単体 | 776MB | 1149MB | 1726MB |
| 60s | 1440 | 同居 | 776MB | 1149MB | 2698MB |

VRAM はクリップ長(240→1440フレーム、6倍)に対して**実質フラット**(torch 側は
774→776MBとほぼ誤差の範囲、nvidia-smi 側は完全に一定)。同居時の nvidia-smi ピークが
単体より一貫して+972MB高いのは BiRefNet+YOLOX の ONNX Runtime CUDA セッション分と
整合(理論値と一致)。**判定**: 60s 同居ピーク 2698MB ≪ 8000MB 予算 → **予算内**。

**スタブの懸念との整合**: 直下のエントリが引用する GitHub issue の「~60GB」報告は
`offload_video_to_cpu`/`offload_state_to_cpu` を使わない(または hiera_tiny 以外の)
構成だったと考えられる。本実測ではこの2フラグを明示的に有効化しており、かつ
`sam2.1_hiera_tiny` のメモリアテンションは(過去フレーム全件ではなく)有限のウィンドウ
のみを参照する設計のため、GPU 常駐分がクリップ長に対して増大しない -- 直下のスタブが
「解決していない」としていた懸念点は、この構成では実際には発生しないことを確認した。

**注意点(既知の制約、B-2 では要検討)**: (1) `init_state` はパス入力のみで mp4 直読み
不可(decord 未導入)-- 本番導入時は現行のフレーム単位 `cv2.VideoCapture` ストリーミング
設計との統合方法を別途検討する必要がある(直下のスタブのもう一つの懸念、これは未解決の
まま)。(2) Triceratops は121フレームしか無くループで水増ししているため、実際の連続
60秒ショットでの挙動(例えば本物の場面転換やシーン内変化が伴う場合のメモリバンク挙動)
は未検証 -- VRAM の「フレーム数に対する挙動」としては妥当な近似だが、追跡品質の検証には
使えない(品質は B-2 のスコープ)。

**次のステップ**: B-2(`build_trimap_alpha` 引数化 + `additive`/`symmetric` プロトタイプ、
D の疑似GTで採否判定)。本コミットのスコープ外。

### B-2: SAM2 シード付きトリマップ合成 プロトタイプ -- Triceratops 実測、判定は **PENDING**(2026-09-23)

第9計画 Part B のステップ2。`tool/pipeline/chroma.py` の `build_trimap_alpha` に
`seed`/`seed_variant`/`band`/`radius_frac` を引数化(既定呼び出しはバイト同一、回帰
テスト `tool/tests/test_chroma_seed_trimap.py` で確認)し、`tool/scripts/
sam2_trimap_prototype.py` でオフラインプロトタイプを実装・実行した。**本コミット時点で
D(生成側のペア素材、単色/自然背景)は未着手のまま届いていない** -- 第9計画 R6 の
指示どおり、以下の数値がどれだけ良く見えても **採否は確定させない**。

**方法**: frame0 の BiRefNet alpha>0.9 の最大連結成分 → `scipy.ndimage.
binary_fill_holes`(この順序が重要 -- 穴を埋める**前**に SAM2 へシードすると、
Triceratops の口内穴欠陥そのものを「背景」として追跡させてしまい、直そうとしている
欠陥をシードに焼き込むことになる)を SAM2 video predictor に `add_new_mask` で
プロンプト、frame0 の IoU(SAM2 mask, alpha>0.5)<0.8 なら bbox プロンプトへ再試行。
トラック消失フレーム(area<0.2×median または IoU<0.3)は BiRefNet 単体 alpha に
自動フォールバック(クリップ全体を通じて記録)。事前登録済み2変種(R7)を
`chroma.build_trimap_alpha` の `seed`/`seed_variant`/`band="alpha"` 経由で合成:
`additive`(本命、FG=erode(seed)|(a_raw>0.90)、BG=~dilate(seed)&(a_raw<0.10) --
SAM2 は BiRefNet の確信 FG を絶対に消せない)と `symmetric`(比較用、FG=erode(seed)、
BG=~dilate(seed))。`mc_median_half` 既定(1)と 0 の両方で計測。SAM2 worker は
追跡完了後すぐ停止し(R8)、BiRefNet/despill の CPU 作業とは同時 GPU 常駐しない。
box_mode_experiment.py と同じノイズ床パターン(baseline を独立に2回推論、A/B の
乖離をノイズ床とする)を採用 -- レバーC で見つかった「ノイズ床=0」は SAM2 という
新しい機構に自動的には持ち越さない前提で、この実行でも実際に0であることを確認した
(下表)。

**実測(Triceratops.mp4、121ネイティブフレーム、`data/sam2_trimap_prototype/run1/
results.json`)**:

| 指標 | baseline(A/B同値) | ノイズ床(A vs B) | additive mh1 | additive mh0 | symmetric mh1 | symmetric mh0 |
|---|---|---|---|---|---|---|
| S1_mc_chatter | 977093 | 0 | 938984 (-3.9%) | 1302196 (**+33.3%**) | 931851 (-4.6%) | 1246089 (**+27.5%**) |
| F3_interior_holes(total) | 152409 | 0 | 146599 (-3.8%) | 145882 (-4.3%) | 147327 (-3.3%) | 146593 (-3.8%) |
| F3 worst_frame | 89 | 0 | 89 | 89 | 89 | 89 |
| F3 worst_value(frame89) | 17741 | 0 | **17741(±0)** | 18714 (**+5.5%**) | **17743(≈±0)** | 18714 (**+5.5%**) |
| wall(baseline_A/B比較用の推論込み総時間) | 154-158s | -- | 155s | 76s | 160s | 78s |

SAM2 追跡診断: frame0 IoU(SAM2 mask, alpha>0.5)=0.9976(bbox 再試行不要)、
トラック消失フォールバック 0/121 フレーム、SAM2 追跡自体の所要時間 40.5s(1クリップに
つき1回、両変種で共有)。F1i/F2 は Triceratops が非クロマのため `None`(既存の
G4 ゲートどおり、想定通り)。

**目視(マゼンタ合成クロップ、口内穴の最悪フレーム89、baseline/additive/symmetric を
同一クロップ範囲で比較)**: `data/sam2_trimap_prototype/run1/crops/Triceratops/`
の3枚(`baseline_A_frame89.png`/`additive_mh1_frame89.png`/`symmetric_mh1_frame89.png`)
を実際に目視した。**3枚とも口内は視覚的に区別がつかない** -- マゼンタが口内に
漏れて見える箇所はどれにも見当たらず、SAM2 シードによる目に見える改善も、目に見える
新規欠陥もない。F3 worst_value がほぼ不変(17741→17741/17743)という数値と目視が
一致しており、これは「指標が見えていないだけ」ではなく、**本当にこのフレームでは
SAM2 シードが口内穴欠陥に対して実質的な効果を持たなかった**ことを示している。
これは当初の仮説(SAM2 の object identity 追跡が口内を FG に含めることで意味的
誤認を消す)に対して**期待外れの結果**であり、好意的に書き換えない: F3 の
クリップ全体の合計は -3.3〜-4.3% 改善しているが、それはこの最悪フレームでの
改善ではなく、他の(軽微な)欠陥フレームでの改善に由来する。

**S1(mc_median の要否)**: `mc_median_half` を既定のまま使う場合、additive/symmetric
とも S1 は baseline よりわずかに改善(-3.9%/-4.6%、ノイズ床0を明確に超える)。
しかし `mc_median_half=0`(mc_median を切る)にすると両変種とも **S1 が baseline
より大幅に悪化**(+27.5%/+33.3%)する上、口内穴の worst_value も悪化(+5.5%)。
第9計画が示唆していた「SAM2 自身の時間整合性が mc_median の代替になり得る」という
仮説は、**この実測ではむしろ逆(SAM2 は mc_median の代わりにならない)**という結果
になった -- mc_median を外す高速化(155s→76s、約半分)の代償が品質悪化として
はっきり出ている。

**判定: 保留(PENDING)**。理由は2つ: (1) D の疑似GT が届いていないため、上記の
数値がどれだけ良くても悪くても「採用/不採用」を確定できない(第9計画 R6 の明示的な
指示どおり)。(2) 口内穴という B レバーの当初の主要な狙いについて、この実測では
効果が確認できなかった -- 良い数字だけを見て採用に傾けることをしない。

#### 追記: flatchroma2 ①感謝(フラットクロマ、`--use-keyer off` で強制ニューラル経路)実測

同じ `data/sam2_trimap_prototype/run1_flatchroma2/results.json` の手順で1本追加実測した
(124フレーム)。SAM2 追跡診断は Triceratops と同様良好: frame0 IoU=0.9949、
フォールバック 0/124。ノイズ床はこのクリップでも実測0(baseline A/B完全一致)。

| 指標 | baseline(A/B同値) | additive mh1 | additive mh0 | symmetric mh1 | symmetric mh0 |
|---|---|---|---|---|---|
| F1i_false_erase_interior | 21321 | 21267 (-0.3%) | 20278 (-4.9%) | 21298 (-0.1%) | 20287 (-4.9%) |
| **F2_false_keep** | 248 | **395 (+59.3%)** | 316 (+27.4%) | **392 (+58.1%)** | 314 (+26.6%) |
| S1_mc_chatter | 98905 | 98855 (-0.1%) | 123833 (**+25.2%**) | 98725 (-0.2%) | 123679 (**+25.1%**) |
| F3_interior_holes(total) | 25417 | 24278 (-4.5%) | 24404 (-4.0%) | 24288 (-4.4%) | 24411 (-4.0%) |
| F3 worst(frame72) | 2657 | 2656 | 2645 | 2657 | 2645 |
| wall | 45.1-45.5s | 45.1s | 21.6s | 44.5s | 20.1s |

**これは正直に報告すべき悪い結果を含む**: F1i/S1 は(mc_median 既定時)ほぼ横ばい
〜わずかに改善、F3 も -4%前後改善しているが、**F2(false_keep、背景を誤って前景に
残す欠陥)がノイズ床0に対し additive/symmetric とも mh1 で +58〜59%、mh0 でも
+27%前後悪化している**。絶対数(248→395など)は小さいが、相対悪化はゲート
「F1i/F2はbaselineを下回らない」に対する明確な**未達**であり、good-newsだけを
拾って書かない。SAM2 のシード幾何(additive の `BG=~dilate(seed)&(a_raw<0.10)`)が
このクリップでは背景側の判定をわずかに甘くしている可能性がある(未検証の仮説、
原因はこのコミットでは特定していない)。mc_median_half=0 では Triceratops 同様に
S1 が大きく悪化(+25%前後)する再現性も確認できた。

**次のステップ**: 残りの flatchroma2(②喜び③思考④挨拶)と cyclorama 4本の回帰確認は
本コミット時点で未着手(GPU 時間の都合、5クリップ+cycloramaのフルセットは別セッション
に持ち越す)。F2 悪化の原因調査(additive のBG判定条件の見直し、または
`bg_seed_thresh`の調整余地)を次のステップに追加する。D の疑似GTが届き次第、
本エントリの数値を `dev/eval_vs_gt.py` の SAD/MSE と突き合わせ、最終的な採否判定を
別エントリに記録する。**現時点でのPENDING判定を裏付ける材料がさらに増えた**
(口内穴の効果なし + F2の悪化)。

#### 追記: B-2 クローズ(2026-09-23) — これ以上の GPU 実験は追わず、保留を確定する

上記2クリップの実測だけで、ablation原則(閾値をいじって延命させない、根拠なく
採用に傾けない)に照らして判断材料は十分そろった: (1) レバーBの当初の主目的
(Triceratopsの口内欠陥修正)は数値・目視とも効果を確認できず、(2) F2が
ノイズ床0に対し明確に悪化(+58〜59%)、(3) D の疑似GT素材は依然として届いていない。
残り flatchroma2 3本・cyclorama 4本を追加実測しても、この2つの否定的所見を覆す可能性は
低く、GPU時間を追加投入する優先度は低いと判断する。**よってB-2はここで正式に
クローズし、「現状不採用・将来再訪の出発点として保留」を本セッションの最終結論と
する。** 残り5クリップの実測・F2原因調査は着手しない。B-3(本線統合)も着手しない。

**F2悪化の原因についての具体的な仮説(コード読解による、実験による検証は未実施)**:
`tool/pipeline/chroma.py:126` の
```python
fg_mask = eroded_seed | (a_raw > fg_seed_thresh)
```
が疑わしい。`additive`・`symmetric`の両変種に共通するのは、`eroded_seed`に含まれる
画素を**`a_raw`の値に関係なく無条件でalpha=1.0に確定させる**という設計。SAM2の
トラッキングマスクは`erode`後も境界付近で数px、BiRefNet自身が「背景寄り」と
判断していた画素(低い`a_raw`)を含んでしまうことがあり、そこが強制的に不透明化
されると背景残存(F2)が増える。両変種でF2悪化がほぼ同率(+58〜59%)なのは、
両者に共通するこの`eroded_seed`強制FG化が主因である可能性が高い(`symmetric`は
`BG=~dilate(seed)`側の条件が`additive`よりさらに緩いにもかかわらずF2悪化率が
ほぼ同じであることは、悪化の主因が`BG`側の緩さではなく`FG`側の強制にあることを
示唆する、という追加の傍証)。

**改善候補(未実装、将来この機能を再訪する場合の出発点として記録)**:
1. `eroded_seed`単独でFGを確定させず、`eroded_seed & (a_raw > 低めの閾値)`のように
   BiRefNet自身の判断も条件に含める(SAM2は「確信を後押しする」だけにし、
   「単独で確定させる」権限は持たせない)
2. SAM2マスクをより強く収縮(erode半径を拡大)させ、境界のはみ出しを抑える
3. 第4回監査のG1(色ベースtrimap)不採用時と同じ教訓 — 「完全上書きでなく
   加重平均」— をUNK帯だけでなくFG/BG判定自体にも適用する(SAM2とBiRefNetの
   投票を確率的に混ぜる)

**本番への影響なし**: `chroma.build_trimap_alpha`の拡張は`seed=None`の既定呼び出しで
バイト同一(回帰テストで保証済み)、`tool/scripts/sam2_trimap_prototype.py`は
サーバー/CLIの本番経路から一切呼ばれないスタンドアロン実験スクリプト。
`config.use_sam2_track`のような新規本番フラグは追加していない。

### SAM2 video-tracking mode

SAM2 ships a video-tracking mode (memory-attention propagation across frames
from a single prompt) that looked like it could replace the per-frame YOLOX
detection + hand-written optical-flow temporal patch on the person route.
Investigated and **not adopted**, for two independent reasons found in SAM2's
own public API and issue tracker:

- `SAM2VideoPredictor.init_state()` only accepts an MP4 path or a directory of
  JPEGs, and loads **every frame into one tensor up front** — there is no
  in-memory/streaming frame API. Adopting it would mean abandoning this
  pipeline's current frame-at-a-time `cv2.VideoCapture` -> ffmpeg streaming
  design in favor of a decode-everything-first one.
- VRAM grows with clip length (the memory bank keeps accumulating
  conditioning/non-conditioning frame features): Meta's own figures put
  `sam2.1_hiera_tiny` at ~4GB VRAM at 1080p, and a GitHub issue reports a
  2m30s/24fps clip using ~60GB before manual memory-bank clearing. On this
  project's 6GB dev GPU (already sharing VRAM with BiRefNet's onnxruntime
  session), that leaves no real headroom for anything but the shortest clips
  without additional memory-management code (SAM2 has no built-in periodic
  memory-bank reset).

Revisit only if a future SAM2 release adds a bounded/streaming memory-bank
mode, or if clip length in practice stays short enough that this stops
mattering.

### Alternative matting models

Web/paper survey for a BiRefNet_lite replacement (permissive-license
candidates only). Benchmarked against this repo's GT (`dev/eval_vs_gt.py`,
SAM2-gate off so it's a pure matte-model comparison) on `sample/giraffe.mp4`
and `sample/widepose.mp4`, 20 frames each, CPU:

- **BEN2** (PramaLLC, MIT, ONNX weights on HF) — box-crop pipeline scored
  giraffe IoU 0.993 / Bnd-F 0.995 (current: 0.994 / 0.992) — on par, but ~1.5-
  1.7x slower per frame on CPU (it's a GPU/fp16-oriented architecture; fp32
  CPU inference has no equivalent fast path). Not adopted: no accuracy edge
  to justify the speed loss.
- **BiRefNet_lite-matting** (ZhengPeng7, MIT, same repo/architecture as
  `birefnet_lite.onnx` but fine-tuned on matting-specific datasets instead of
  generic segmentation) — no ONNX weights ship on HF, so it had to be
  self-exported from `model.safetensors` (torch 1.13.1 + a
  `deform_conv2d_onnx_exporter` patch; torch>=2.0's exporter doesn't support
  this model's `ASPPDeformable` blocks). Export succeeded and the resulting
  ONNX runs correctly under onnxruntime, but scored giraffe IoU 0.994 /
  Bnd-F 0.993 and widepose IoU 0.948 / Bnd-F 0.970 vs. current 0.994/0.992 and
  0.951/0.970 — within noise, no consistent edge either direction. Not
  adopted: matting-specific fine-tuning didn't measurably help on these
  clips.

Revisit either only with a larger/more diverse GT set, or if a future release
of either model changes materially.

### CPU-side region gate & other speed levers

- **EfficientViT-SAM-L0** (Apache-2.0, MIT-Han-Lab, ONNX on HF) as a CPU
  substitute for SAM2's region gate — a published benchmark (arXiv 2410.04960)
  reports ~194ms/frame on a server CPU, which looked like it could finally
  make a SAM2-equivalent gate viable on `--device cpu`. Self-benchmarked on
  this project's dev CPU: ~1.8-2.7s/frame (slower than the paper's number on
  this weaker CPU, but still well under a "still usable" bar) — however,
  gating BiRefNet's alpha with its box-prompted mask (same combination style
  as the GPU BiRefNet+SAM2 path) **hurt** GT accuracy: giraffe IoU
  0.994->0.893, widepose IoU 0.951->0.792. The mask is too coarse (box-prompt
  only, no fine hair/fur detail) to gate a already-high-precision BiRefNet
  alpha without doing more harm than the clutter it would have removed. Not
  adopted — also note, undocumented upstream: the ONNX decoder expects box
  coordinates in *original* image pixel scale, not pre-scaled to the 512
  encoder input (pre-scaling silently produces a near-empty mask).
- **BiRefNet_lite.onnx dynamic INT8 quantization** (onnxruntime's own
  `quantize_dynamic`) — full quantization fails outright (`ConvInteger` has no
  CPU-EP kernel in this onnxruntime build); restricting to `MatMul` layers
  only (BiRefNet is Conv/Swin-backbone-heavy, so this is most of the model
  left untouched) succeeded but gave inconsistent speed (giraffe 11.6s->15.3s
  *slower*, widepose 12.1s->10.8s faster) with no accuracy change. Not adopted
  — no reliable win.
- **`cv2.ximgproc.guidedFilter`** in place of the hand-rolled box-filter
  guided filter — would require adding `opencv-contrib-python-headless` as a
  new dependency (`ximgproc` isn't in plain `opencv-python-headless`). Skipped
  before benchmarking: not worth a new dependency for what research suggested
  would be a same-algorithm, marginal-speed swap.
- **DirectML** (as a way to sidestep the onnxruntime-CUDA/torch-CUDA
  contention entirely) — `torch-directml` has no working SAM2 port (open,
  unanswered upstream issue), and onnxruntime's own DirectML EP is reported
  1.5-2.8x *slower* than CUDA EP for transformer models on Windows. Dead end,
  not prototyped.

### Automatic GPU/CPU device selection (2026-09-14)

`server/presets.py`'s `build_config` now defaults `device` to `"auto"` and
resolves it through `tool.matte_core.resolve_device`, which checks whether
onnxruntime and torch can actually reach CUDA rather than assuming they can.
Previously the server always requested `"cuda"` and, if the CUDA stack was
unreachable, onnxruntime silently fell back to CPU with only a RuntimeWarning
in the log -- a 50s/frame run that looked identical to a healthy one from
outside. The status strip also gained a GPU/CPU toggle that writes an explicit
override (clicking the pressed side clears it, returning to auto).

**Caught in review: this introduced a 7-second stall on the first job.**
`resolve_device` imports torch and calls `torch.cuda.is_available()`, ~7s
cold, and as the new default it ran inside the `POST /api/jobs` handler -- so
the first submission after every restart hung. It surfaced only as a failing
E2E timing test ("running shows a live preview mid-flight"), which looked like
the pre-existing flake in that suite; confirming it by stashing the changes and
re-running against HEAD (2/2 pass) versus with them (3/3 fail) is what
separated the two. Now memoized (`presets.resolved_auto_device`, the answer
cannot change mid-process) and warmed in `app.py`'s lifespan, with a
regression test asserting the probe runs at most once per process.

## Researched, not yet implemented (candidates for future work)

A wider lateral-thinking pass beyond model/backend swaps, each only
web/design-researched (not prototyped):

- **Multi-subject support** — today `subject_box()` picks a single largest
  box; everything else in frame becomes background. Detection/SAM2 changes
  are cheap (YOLOX already returns every box; SAM2's encoder runs once, the
  decoder is cheap per extra box-prompt), but BiRefNet has no permissive
  joint multi-instance mode, so matting cost scales ~N×. Image mode is a
  natural incremental extension; video mode would need a real cross-frame
  instance-identity/tracking component this pipeline has never needed before
  (single-subject "biggest box" is trivially stable frame-to-frame; N
  subjects is a genuine multi-object-tracking problem) — a partial
  rearchitecture, not just more compute.
- **Broader GT/eval basis** — all quality decisions so far rest on 2 hand-
  annotated clips. DIS5K/P3M-10k/AM-2k are candidate GT sources but are
  research-use-licensed for the *dataset*, not permissive — usable only as
  gitignored internal eval data, never redistributed in this repo (see
  [`docs/LICENSE_POLICY.md`](LICENSE_POLICY.md)). SAD/Grad (Deep Image
  Matting's metrics) are cheap additions to `dev/eval_vs_gt.py` (SAD is
  already in `pymatting`); LPIPS/SSIM were judged not worth the dependency.
  ~6-8 more clips (fur close-up, motion blur, multi-subject,
  semi-transparent/reflective object, low-contrast/backlit, second furry
  animal, one anime/illustration clip to finally validate that fallback
  route) would meaningfully raise confidence without becoming a full
  benchmark suite.
- **Cross-platform**: Linux is the highest-confidence win *not yet
  testable from this Windows dev machine* — the WDDM driver model (not CUDA
  itself) is the likely root cause of both the onnxruntime/torch
  cross-context contention this project fought hard, and the DLL-path pain
  with `onnxruntime-gpu`; Linux may need neither workaround. SAM2->CoreML
  already has official Apple artifacts (near-free). BiRefNet->CoreML hits
  the same `deform_conv2d` wall already hit exporting to plain ONNX, but a
  community workaround exists (op substitution, like this project's own
  ONNX export patch). NPU (OpenVINO/QNN) has no evidence yet of anyone
  running a Swin/ViT model like this successfully — likely hits the same
  operator-support wall, lowest priority of the three.
- **Input-side efficiency**: `BiRefNet_dynamic` (MIT, trained 256-2304px) could
  let small subjects run below the fixed 1024x1024 cost, but no published
  accuracy-vs-resolution ablation exists — would need this project's own
  measurement before trusting it. Alpha-specific frame interpolation (RIFE/
  FILM, both permissive) to reduce the person route's per-frame cost is
  technically fine license-wise but has no established precedent — these
  nets hallucinate photorealistic RGB texture, not preserve segmentation
  boundaries; unproven, second-tier experiment at best.
- **Operational throughput**: GPU batch inference is a **non-starter** on
  this project's 6GB Turing GPU (BiRefNet alone needs ~5.5GB at batch=1; no
  Tensor Cores to make batching pay off anyway) — don't build it. CPU
  multi-file parallelism (`ProcessPoolExecutor`, one worker per file) is a
  real, standard win, but only with `intra_op_num_threads` explicitly capped
  per worker (uncapped, onnxruntime's own internal threading contends with
  process-level parallelism and can be *slower* than sequential). Video
  resume/checkpointing is over-engineering for this project's actual clip
  lengths (a few hundred frames) — skip unless clips get much longer.
- **Self-improvement**: fine-tuning BiRefNet/YOLOX on this project's own
  handful of annotated frames was judged a bad fit — the target domain here
  is broad/general-purpose, and fine-tuning on a narrow, highly-correlated
  clip set risks exactly the kind of out-of-domain regression seen in
  comparable published ablations (in-domain gain, out-of-domain accuracy
  *drop*), with no distinct validation content to even catch it. Test-time
  augmentation (flip/multi-scale) gives a marginal, unquantified-for-
  BiRefNet accuracy bump at 2-4x inference cost — not worth it as a
  pipeline-wide default on an already CPU-slow path.
