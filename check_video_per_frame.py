import cv2


def check_frame(path):
    frame_num = 0
    vid = cv2.VideoCapture(path)
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            print(return_value)
            cv2.imshow('output video', frame)
            cv2.waitKey(0)
        else:
            break


if __name__ == "__main__":
    path = ''
    check_frame(path)