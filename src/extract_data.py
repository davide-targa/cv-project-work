from pathlib import Path


def get_image_data():
    image_data = []
    for idx, file_path in enumerate(sorted(Path("src/data/PennFudanPed/Annotation/").glob("*.txt")), start=1):
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.readlines()
            for line in content:
                if line.startswith("Image filename :"):
                    filename = Path(line.split(":")[1].strip().strip('"')).name

                if line.startswith("Bounding box for object"):
                    object_index = int(line.split(" ")[4])
                    coords = line.split(": ")[1].strip().split(" - ")
                    x_min = int(coords[0].split(", ")[0].lstrip("("))
                    y_min = int(coords[0].split(", ")[1].rstrip(")"))
                    x_max = int(coords[1].split(", ")[0].lstrip("("))
                    y_max = int(coords[1].split(", ")[1].rstrip(")"))
                    image_data.append(
                        {
                            "filename": filename,
                            "object_index": object_index,
                            "x_min": x_min,
                            "y_min": y_min,
                            "x_max": x_max,
                            "y_max": y_max,
                        }
                    )
                    print(f"{idx} - {filename:<16} ({object_index}): ({x_min}, {y_min}) - ({x_max}, {y_max})")

    print(f"Found {len(image_data)} objects in total.")
    print(image_data)
    return image_data


if __name__ == "__main__":
    get_image_data()
