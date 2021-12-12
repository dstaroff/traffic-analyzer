from cv2 import cv2

from src.detector.vehicle.detector import VehicleDetector
from src.utils import const

detector = VehicleDetector()

radius = 5


def process(frame):
    vehicles = detector.detect(frame)

    for vehicle in vehicles:
        cv2.circle(
                frame,
                (int(vehicle.centroid().x), int(vehicle.centroid().y)),
                radius,
                vehicle.color(),
                0,
                )

        cv2.putText(
                frame,
                f'{vehicle.caption()}: {vehicle.score():.3f}',
                (int(vehicle.centroid().x) - 2 * radius, int(vehicle.centroid().y) - 2 * radius),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                vehicle.color(),
                1,
                )

    #
    # for i in range(0, r["rois"].shape[0]):
    #     # extract the class ID and mask for the current detection, then
    #     # grab the color to visualize the mask (in BGR format)
    #     classID = r["class_ids"][i]
    #     mask = r["masks"][:, :, i]
    #
    #     vehicle_type = vehicle_from_class_id(classID)
    #     if vehicle_type is not None:
    #         frame = visualize.apply_mask(frame, mask, vehicle_type._color)
    #
    # # loop over the predicted scores and class labels
    # for i in range(0, len(r["scores"])):
    #     # extract the bounding box information, class ID, label, predicted
    #     # probability, and visualization color
    #     (startY, startX, endY, endX) = r["rois"][i]
    #     classID = r["class_ids"][i]
    #     score = r["scores"][i]
    #
    #     vehicle_type = vehicle_from_class_id(classID)
    #     if vehicle_type is not None:
    #         # cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
    #         y = startY - 10 if startY - 10 > 10 else startY + 10
    #         cv2.putText(
    #                 frame,
    #                 f'{vehicle_type._caption}: {score:.3f}',
    #                 (startX, y),
    #                 cv2.FONT_HERSHEY_SIMPLEX,
    #                 0.5,
    #                 (vehicle_type._color[2], vehicle_type._color[1], vehicle_type._color[0]),
    #                 1,
    #                 )

    cv2.imshow(const.APP_NAME, frame)


def process_video(file_path):
    cap = cv2.VideoCapture(file_path)

    while True:
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        ret, frame = cap.read()
        if ret:
            process(frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()


def main():
    process_video('.\\video\\04.mp4')
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
