from common import log
from PIL import Image, ImageDraw


__all__ = ['line', 'Image', 'ImageDraw', 'black', 'white', 'red', 'green']


black = (0, 0, 0, 255)
white = (255, 255, 255, 255)
red = (255, 0, 0, 255)
green = (0, 255, 0, 255)
blue = (0, 0, 255, 0)


def _line(x0, y0, x1, y1, image, color):
    xy = []  # collect xy coordinates

    # fix holes by choose max(dx, dy) as step
    steep = False
    if abs(x0 - x1) < abs(y0 - y1):  # transpose if steep
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        steep = True

    # make indepent on order of points
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    dx = x1 - x0
    dy = y1 - y0
    derror2 = abs(dy) * 2
    error2 = 0
    y = y0

    for x in range(x0, x1 + 1):
        # t = (x - x0) / (float)(x1 - x0)
        # y = int(y0 * (1. - t) + y1 * t)
        if steep:
            xy.append(y)
            xy.append(x)
        else:
            xy.append(x)
            xy.append(y)
        
        error2 += derror2
        if error2 > dx:
            y += 1 if y1 > y0 else -1
            error2 -= dx * 2

    # send xy sequence && set color
    ImageDraw.Draw(image).point(xy, color)


def line(*arg):
    if len(arg) == 4:
        # line(v0, v1, image, color)
        v0, v1, image, color = arg
        _line(v0.x, v0.y, v1.x, v1.y, image, color)
    elif len(arg) == 6:
        # line(x0, y0, x1, y1, image, color)
        _line(*arg)
    else:
        raise ValueError('arg numbers must be 4 or 6.')


def main():
    image = Image.new('RGBA', (100, 100), black)
    line(13, 20, 80, 40, image, white)
    line(20, 13, 40, 80, image, red)
    line(80, 40, 13, 20, image, red)
    output = image.transpose(Image.FLIP_TOP_BOTTOM)
    output.save('./output/line.png')

if __name__ == '__main__':
    main()
