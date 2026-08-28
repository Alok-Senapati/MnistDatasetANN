"""Convert the IDX-format MNIST files into CSV files for the project."""

from __future__ import annotations


def convert(image_file: str, label_file: str, output_file: str, sample_count: int) -> None:
    """Convert a single MNIST split from binary IDX format to CSV rows.

    The data files should be located in the current working directory when this helper
    is invoked. The output is a row-oriented CSV where the first column contains the
    numeric label and the remaining 784 columns contain the grayscale pixels.
    """
    with (
        open(image_file, "rb") as image_stream,
        open(label_file, "rb") as label_stream,
        open(output_file, "w", encoding="utf-8", newline="") as output_stream,
    ):
        image_stream.read(16)
        label_stream.read(8)
        images: list[list[int]] = []

        for _ in range(sample_count):
            image = [ord(label_stream.read(1))]
            for _ in range(28 * 28):
                image.append(ord(image_stream.read(1)))
            images.append(image)

        for image in images:
            output_stream.write(",".join(str(pixel) for pixel in image) + "\n")


convert("train-images.idx3-ubyte", "train-labels.idx1-ubyte", "mnist_train.csv", 60000)
convert("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", "mnist_test.csv", 10000)
