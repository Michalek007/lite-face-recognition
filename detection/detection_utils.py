from PIL import Image, ImageDraw


def display_boxes(image: Image, boxes, true_boxes=None, save_image: bool = False, filename: str = 'image_with_boxes.jpg'):
    line_width = int(image.width*0.01)
    if not line_width:
        line_width = 1
    for x, y, x2, y2, prob in boxes:
        draw = ImageDraw.Draw(image)
        draw.rectangle((x.item(), y.item(), x2.item(), y2.item()), outline='red', width=line_width)
    if true_boxes is not None:
        for x, y, x2, y2 in true_boxes:
            draw = ImageDraw.Draw(image)
            draw.rectangle((x.item(), y.item(), x2.item(), y2.item()), outline='green', width=line_width)
    if save_image:
        image.save(filename)
    else:
        image.show()
