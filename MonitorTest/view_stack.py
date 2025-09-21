import cvzone
from cvzone.PlotModule import LivePlot

class ViewStacker:
    def __init__(self, width=400, height=600, y_range=(25, 40)):
        self.plotY = LivePlot(width, height, list(y_range))

    def stack(self, img, ratioAvg, color):
        imgPlot = self.plotY.update(ratioAvg, color)
        return cvzone.stackImages([img, imgPlot], 2, 1)