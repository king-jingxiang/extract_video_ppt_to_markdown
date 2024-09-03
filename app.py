import gradio as gr

import video_extract as ext


def on_image_click(evt: gr.SelectData, video_file):
    # evt.index 是点击的图片的索引
    selected_index = evt.index
    print(f"get_summary_content_with_index with video_file {video_file}, selected_index {selected_index}")
    return ext.get_summary_content_with_index(video_file, selected_index)


# Gradio界面
with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.TabItem("视频PPT提取-命令式"):
            with gr.Row():
                gr.Markdown("## 请先上传视频或者指定本地视频文件路径")
            with gr.Row():
                video_input = gr.Video(label="上传视频")
            with gr.Row():
                threshold = gr.Slider(minimum=1, maximum=10, step=1, value=5, label="图片相似度阈值")
                interval_sec = gr.Slider(minimum=0.1, maximum=10, step=0.5, value=1, label="抽帧间隔（秒）")
                merge_size_threshold = gr.Slider(minimum=0, maximum=4096, step=1, value=512, label="字幕合并阈值(B)")
            with gr.Row():
                extract_frame_button = gr.Button("提取并总结ppt")
            with gr.Row():
                gr.Markdown("## 输出结果展示")
            with gr.Row():
                frame_images = gr.Gallery(
                    label="提取的相似帧", show_label=False, elem_id="gallery"
                    , columns=[1], rows=[1], object_fit="contain", height="auto")
                summary_md_output = gr.Markdown()
            with gr.Row():
                gr.Markdown("## 文件输出列表")
            with gr.Row():
                file_output = gr.File(label="输出文件", show_label=False, file_types=["zip"])

            # 添加示例
            examples = gr.Examples(
                examples=[
                    ["videos/example_video1.mp4", 5, 1, 128],
                ],
                inputs=[video_input, threshold, interval_sec, merge_size_threshold]
            )

            extract_frame_button.click(fn=ext.extract_and_summary_video,
                                       inputs=[video_input, interval_sec, threshold, merge_size_threshold],
                                       outputs=[frame_images, file_output])
            frame_images.select(on_image_click, inputs=[video_input], outputs=summary_md_output)
        with gr.TabItem("视频PPT提取-对话式(开发中)"):
            pass

# 限制并发为1，防止显存溢出
demo.queue(max_size=1)
demo.launch(server_name="0.0.0.0", )
