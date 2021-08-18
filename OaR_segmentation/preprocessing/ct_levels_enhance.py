# channels agnostic
def setDicomWinWidthWinCenter(img_data, winwidth, wincenter):
    img_temp = img_data
    min = (2 * wincenter - winwidth) / 2.0 + 0.5
    max = (2 * wincenter + winwidth) / 2.0 + 0.5
    dFactor = 255.0 / (max - min)

    img_temp = (img_temp - min) * dFactor

    img_temp[img_temp < 0] = 0
    img_temp[img_temp > 255] = 255
    return img_temp