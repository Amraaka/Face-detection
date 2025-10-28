from cvzone.FaceMeshModule import FaceMeshDetector

class FaceMeshService:
    def __init__(self, max_faces=1):
        self.detector = FaceMeshDetector(maxFaces=max_faces)

    def find_face(self, img, draw=False):
        return self.detector.findFaceMesh(img, draw=draw)