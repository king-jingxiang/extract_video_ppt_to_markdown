# extract_video_ppt_to_markdown

#### 介绍

* 视频内容提取： 自动提取视频中的PPT图片，帮助学习者快速定位关键视觉信息。
* 音频转字幕： 使用ASR模型将视频音频转换为字幕，增强内容的可访问性。
* 字幕总结： 利用大语言模型对字幕进行总结，生成结构化的Markdown文件，便于复习和资料整理。

### 兼容国产算力：
这两个模型可以正常运行在天数智芯天垓100加速器上。

### 性能
应用运行所需的算力：
处理25分钟视频大约需要6分钟，表明该应用对算力的需求较高。特别是对于大型模型如Whisper-large-v3和Qwen2-7B-Instruct，需要较强的计算资源和推理优化来保证处理速度。

### 响应速度：
当前性能为25分钟视频处理需要6分钟，响应速度相对较慢，但有优化空间。特别是对于实时处理需求，目前的推理延迟较高，不适合在线实时处理。

### 技术挑战

- Token上下文限制： 大型语言模型如Qwen2-7B-Instruct在处理长文本时会受到token上下文限制，导致无法处理过长的视频字幕片段。这需要通过分段处理、上下文拼接等技术手段来解决。
- 显存容量限制： 处理大型模型时，显存容量可能成为瓶颈，特别是在多任务并发处理时。需要通过模型压缩、量化、分布式计算等技术来优化显存使用。
- 推理延迟： 当前推理延迟较高，不适合在线实时处理。需要通过模型优化、硬件加速、并行计算等手段来降低延迟。
- 模型泛化能力： 模型需要适应不同类型和质量的视频内容，这要求模型具有较强的泛化能力和鲁棒性。
- 网络延迟：网络延迟较高，稳定性查，在测试过程中没有本地测试更便捷

### 使用的模型：
- ASR模型： OpenAI/Whisper-large-v3
- LLM模型： Qwen/Qwen2-7B-Instruct

### 进阶版源码

https://gitee.com/qq764073542/extract_video_frames