img2dataset \
--url_list "laion-coco-1M-metadata.tsv" \
--input_format "tsv" \
--url_col "URL" \
--caption_col "TEXT" \
--output_folder "laion-coco-1M-images" \
--thread_count 64 \
--image_size 256