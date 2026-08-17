# Chat TE

ComfyUIの Qwen-Image-Edit 用テキストエンコーダー(Qwen2.5-VL-7B、Qwen3-VL-4B/8B)を、**別のLLMを追加ロードすることなく**そのままチャット(マルチターン・vision対応)として使えるようにするComfyUIカスタムノードです。

- **VRAM節約**: チャット専用に別のLLMをロードしません。ComfyUIが既にロードしているQwen2.5-VL/Qwen3-VLのCLIPをそのまま使い回します。
- **段階的な画像生成**: ラフ案について会話しながらプロンプトを練り、そのままチャットから画像編集/生成ノードのプロンプト欄に送り込めます。

## 仕組み

`comfy.sd.load_clip(..., clip_type=CLIPType.QWEN_IMAGE)`でロードされたCLIPオブジェクトが持つ、ComfyUIネイティブの`CLIP.tokenize()` / `CLIP.generate()` / `CLIP.decode()`をそのまま呼び出しているだけです。モデルの再構築や重みのコピーは一切行っていません。

## 必要環境

- **比較的新しいComfyUI**(`CLIP.generate()`によるテキスト生成のネイティブサポートが必要。動作確認: ComfyUI 0.33.1 / commit `72865f4` / 2026-08-13)。古いComfyUIでは`clip.generate`メソッド自体が存在せず動作しません。
- **Qwen2.5-VL-7B-Instruct**または**Qwen3-VL-4B/8B**のテキストエンコーダーファイル(`models/text_encoders/`配下)。**Qwen2.5-VLの3B・32B・72B、Qwen3(非VL)、Qwen3-VL-32Bは非対応**です(サイズ・アーキテクチャの不一致、またはComfyUI本体が汎用チャット用のネイティブ生成経路を持たないため)。`.safetensors`形式のほか、**`.gguf`形式にも対応**しています(下記「GGUF形式について」参照)。
- 追加のPython依存パッケージは不要です(ComfyUI同梱のものだけで動作します)。GGUF形式のテキストエンコーダーを使う場合のみ、`ComfyUI-GGUF`(city96)および/または`gguf`(calcuis)カスタムノードが別途必要です。

### GGUF形式について

`.gguf`形式のテキストエンコーダーもロードできます。`ComfyUI-GGUF`(city96)または`gguf`(calcuis)がインストールされていれば、パネルのモデル選択プルダウンに`.gguf`ファイルも表示されます。

- **必須**: [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)(city96)または[gguf](https://github.com/calcuis/gguf)(calcuis)のインストール(内部実装はどちらのノードもComfyUIのノードレジストリ経由で呼び出しているだけで、Chat TE自体はGGUFの読み込みロジックを持ちません)。両方インストールされていても構いません(下記の通り自動で使い分けます)。
- **vision(mmproj)の扱い**: テキストと画像エンコーダーが別ファイルに分かれている場合(例: `Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf` + `Qwen2.5-VL-7B-Instruct-mmproj-BF16.gguf`)、同じフォルダに両方置いておけば自動的にファイル名からマッチングしてマージされます。ファイル名の一致条件は「量子化サフィックス(`-q4_k_m`等)を除いた部分がmmprojファイル名に含まれること」なので、命名パターンが近ければ量子化違いのテキストエンコーダーとも意図せずマッチすることがあります。
  - **Qwen2.5-VL**: `ComfyUI-GGUF`(city96)がmmprojの自動マージに対応しています。
  - **Qwen3-VL**: `ComfyUI-GGUF`(city96)はQwen3-VLアーキテクチャのmmproj自動マージに対応していません。`gguf`(calcuis)がインストールされていればそちらへ自動的にフォールバックし、vision込みでロードされます。未インストールの場合はテキストのみのロードになります(ロード自体はエラーになりません)。
- **グラフノード経由は非対応**: `Qwen Chat Bridge (Register CLIP)`ノードにGGUF版のCLIPを直接接続した場合、lm_head補完(下記「既知の制限事項」参照)が働きません。パネルからのロードを使ってください。

## インストール

このフォルダをComfyUIの`custom_nodes/`配下に配置(またはシンボリックリンク/ジャンクション)し、ComfyUIを再起動してください。

## 使い方

1. ComfyUI左サイドバーの「Chat TE」タブを開く
2. ヘッダー右の⚙(設定)アイコンをクリックして設定パネルを開き、モデル選択のプルダウンからテキストエンコーダーファイルを選んで「Load」をクリック(ロード状態はヘッダーの丸印の色でも常に確認できます: 灰=未ロード/緑=ロード済み/赤=エラー)
   - **Thinking mode**(Qwen3-VLのみ有効。Qwen2.5-VLでは無視されます): デフォルトON(モデルの自然な挙動のまま何も手を加えません)。OFFにすると空の`<think></think>`ブロックを付加して思考過程の出力を抑制し即答させますが、モデルによっては(特にlm_headを共有=tied embeddingsな軽量モデル、例: Qwen3-VL-4B)出力が崩れることを実機で確認しているため、基本的にはONのままにしてください。
   - **Max tokens**: 1回の応答で生成する最大トークン数(デフォルト512)。
   - システムプロンプトはブラウザの`localStorage`に保存され、ComfyUIやパネルを再読み込みしても保持されます。
3. チャット欄で会話。画像はドロップゾーンにドラッグ&ドロップで添付でき、vision付きの質問ができます。複数ターンの文脈は自動的に保持されます。ドロップした画像はチャットへの添付と同時にComfyUIの`input/`フォルダにもアップロードされるので、同じ画像をワークフロー側で使うのに再度ドロップし直す必要はありません(下記5.)。添付画像のプレビューと「Send to workflow」ボタンは、メッセージを送信しても消えません。ユーザーが「×」で明示的にクリアするか、新しい画像を添付するまで表示され続けるので、チャットで内容を練ってからワークフローに送る、という順序でも操作できます。
4. キャンバス上でプロンプトを書き込みたいノード(例: `QwenVLTextEncoder`など、`prompt`/`text`/`user_prompt`という名前のウィジェットを持つノード)を選択した状態で、アシスタントの返答にある「Send to encoder」ボタンを押すと、その内容がウィジェットに書き込まれます
5. 画像を添付した状態でキャンバス上の`LoadImage`など`image`という名前のウィジェットを持つノードを選択し、添付プレビューの「Send to workflow」ボタンを押すと、アップロード済みのファイル名がそのウィジェットに書き込まれます。チャットへの添付とワークフローへの読み込みを、1回のドロップだけで済ませられます。
6. あとは通常通りワークフローを実行するだけです

### グラフノードから使う(推奨: 真のVRAM共有を狙う場合)

`Qwen Chat Bridge (Register CLIP)`ノード(`CLIP → CLIP`のパススルー)をワークフロー内の`CLIPLoader`直後に挿入すると、そのCLIPをチャットバックエンドとして登録できます。**この構成で、かつ画像生成ノード側にも同じCLIPLoaderの出力を配線している場合のみ**、`load_clip()`の呼び出しが1回に統一され、チャットと画像生成が完全に同一の重みを共有します(退避・再ロードのコストも二重のVRAM消費も発生しません)。

パネルからモデルを直接ロードした場合や、ワークフロー側が独自に(別途)同じ/別のテキストエンコーダーをロードする場合は、ComfyUI標準のVRAM退避機能によりOOMは回避されますが、チャットと生成を切り替えるたびに再ロードが発生し多少の遅延があります。

### サンプルワークフロー

[`example_workflows/chat_te_sample_qwen_image_edit.json`](example_workflows/chat_te_sample_qwen_image_edit.json)に、上記「グラフノードから使う」構成(`CLIPLoader` → `Qwen Chat Bridge (Register CLIP)` → `TextEncodeQwenImageEdit` ×2)をそのまま体現したサンプルを同梱しています。ComfyUIの「Workflow > Open」からこのファイルを読み込むと、Chat TEパネルからの「Send to encoder」でプロンプトを送り込みつつ、Qwen-Image-Editによる画像編集をVRAM共有ありで試せます。Qwen2.5-VL-7B・Qwen-Image-Edit・関連LoRAのモデルファイルは別途用意してください。

## OpenAI互換API

- `POST /v1/chat/completions`、`GET /v1/models` を、ComfyUI自身のポート上に直接公開しています(別プロセス・別ポート不要)。
- ペイロード形式はOpenAI Chat Completions互換(`messages`、`content`は文字列または`[{type:"text"}, {type:"image_url"}]`のブロック配列、`tools`/`tool_choice`にも対応)です。ストリーミング(`stream: true`)は非対応で、常に単一のJSONレスポンスを返します。
- 外部ツールをこのエンドポイントに向けることができます。[ComfyUI-Workflow-Studio](https://github.com/ketle-man/ComfyUI-Workflow-Studio)のAI TOOLタブが「LM Studio」/「Lemonade」バックエンドとして送るリクエスト形式(エンドポイント・非ストリーミング・画像添付・tool calling)は、いずれもコードレベルの比較でこのAPIと一致することを確認済みです。Backend URLをこのComfyUIのアドレスに向けるだけで追加実装なしに接続できる見込みですが、実際に2つのインスタンスを繋いだ動作確認はまだ行っていません。
- プロンプトに埋め込む前に、Qwenの特殊トークン(`<|im_start|>`等)がユーザー入力・ツール結果に含まれていても無害化するサニタイズ処理が入っています(ロール偽装によるプロンプトインジェクション対策、`py/chat_template.py`の`_sanitize`)。

## 既知の制限事項

- 対応アーキテクチャはQwen2.5-VL-**7B**とQwen3-VL-**4B/8B**のみです。Qwen2.5-VLの3B/32B/72B、Qwen3(非VL、テキストのみ)、Qwen3-VL-32Bは非対応です(Qwen3非VLはComfyUI本体が汎用チャット用の生成経路を持たず、画像生成モデル専用のテキストエンコーダー実装に固定でルーティングされるため)。
- 一部のファイルはファイル名と実際のモデルサイズが一致していない場合があります(例: ファイル名に`fp16`とだけ書かれていて7B版のように見えても、中身がQwen2.5-VL-3Bだった実例あり)。ロード時に「Could not find a supported Qwen language model inside this CLIP」というエラーが出た場合は、まずファイルの実体が本当に対応サイズかを疑ってください。
- パネルからの独立ロード、またはワークフロー側の独自ロードとの組み合わせでは、真のゼロコストVRAM共有にはなりません(上記「グラフノードから使う」を参照)。
- GGUF形式のロードには`ComfyUI-GGUF`(city96)または`gguf`(calcuis)が必要です。未インストールの場合は`.gguf`ファイルを選択してもロード時にエラーメッセージが表示されます。
- tool calling(OpenAIの`tools`/`tool_choice`、Qwenの`<tool_call>`/`<tool_response>`タグ形式)は実装済みで、単発呼び出し・複数呼び出しの並列実行・複数ターンにわたる往復のいずれも実地で動作確認済みです。
- チャット応答の冒頭に稀に`addCriterion`のような余分なトークンが混入することがあります。原因未調査です。

## 謝辞

このプロジェクトは以下のオープンソースプロジェクトの実装・API・調査で得た知見に支えられています。

- **[ComfyUI](https://github.com/comfyanonymous/ComfyUI)**(comfyanonymous, GPL-3.0) — 本ノードの核となる`CLIP.tokenize()` / `CLIP.generate()` / `CLIP.decode()`というネイティブ生成APIそのもの。Chat TEはこのAPI呼び出しに徹し、ソースへの変更は一切加えていません。
- **[ComfyUI-QwenImageWanBridge](https://github.com/fblissjr/ComfyUI-QwenImageWanBridge)**(fblissjr, MIT) — 初期調査でComfyUIロード済みCLIPからテキスト生成する手法の実証を参考にしました。コードはコピーしていません。
- **[ComfyUI-Workflow-Studio](https://github.com/ketle-man/ComfyUI-Workflow-Studio)**(statsu, MIT) — サイドバーパネルのUI設計(チャット・vision添付・システムプロンプト)の参考にしました。
- **[ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)**(city96, Apache-2.0) — GGUF形式のテキストエンコーダー読み込み(Qwen2.5-VL)。Chat TE自体はGGUFの読み込みロジックを持たず、このノードをComfyUIのノードレジストリ経由で呼び出しています。
- **[gguf](https://github.com/calcuis/gguf)**(calcuis, MIT) — 同じくGGUF読み込み。Qwen3-VLアーキテクチャのvision(mmproj)自動マージに対応しているため、Qwen3-VLのGGUF読み込み時にはこちらを利用します。

## ライセンス

MIT。詳細は[LICENSE](LICENSE)を参照してください。

ComfyUI本体はGPL-3.0ですが、`chat_TE`はComfyUI本体のソースファイルを一切変更せず、`import comfy.sd`・`PromptServer.instance`などの公開APIを呼び出しているだけです。custom_nodes向けの拡張機構を介した別プログラムとして扱う、ComfyUIエコシステム全体(MIT/Apache系のカスタムノードが大多数)で採用されている標準的な慣行に沿っています。GGUF読み込みで利用する`ComfyUI-GGUF`(city96, Apache-2.0)・`gguf`(calcuis, MIT)についても、ソースコードは一切コピーせず、ComfyUIのノードレジストリ経由で公開メソッドを呼び出しているだけです。
