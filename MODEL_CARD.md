# Trained BERT sentiment classifier

Base model: bert-base-uncased, revision 86b5e0934494bd15c9632b12f734a8a67f723594. Labels: 0 negative, 1 positive. Fine-tuned on 3,500 training tweets from consecutive source rows 797500–802499; validation/test 750/750; seed 42; three epochs; FP32 AdamW; batch 8; sequence length 128; learning rate 2e-5. Final test accuracy: 0.844 (633/750).

This is the final HW4 reference checkpoint, not the AMP throughput configuration. Validation loss was lowest at epoch 1; the report discloses the cost of continuing through epoch 3. Test metrics in the report use the final checkpoint consistently. Twitter labels are weak supervision and short texts may be ambiguous.

The actual trained FP32 checkpoint and tokenizer are included in the ZIP under `corrected/NVIDIA_GeForce_RTX_4090/checkpoints/hw4_corrected_final`. Model SHA-256: `ed64cb7adb2e8a8c610fafbb24aca75e7b28b0ff882905a0d793ae20b2fe2c61`. The large weights are excluded from Git history, but their checksum is committed in the manifest and they are included in this upload bundle.

Run `python infer.py "I enjoyed this lab."` after installing the pinned Python dependencies. The model loads from local files; no GPU or Hugging Face download is required for this CPU inference path.
