IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
BATCH_SIZE = 64
CLASS = ["QuangHieu", "Hue", "Nhung"]
CLASS_DIST = dict()
for count, i in enumerate(CLASS):
    CLASS_DIST[i] = int(count)
