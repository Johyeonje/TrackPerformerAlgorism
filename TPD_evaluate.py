import numpy as np
import pandas as pd


def evaluate_TPD(path):
  xlsx = pd.read_excel(path, header=1)
  score = 0.
  frame_size = xlsx.shape[0]

  for t in range(frame_size):
    det_obj = xlsx.shape[1]
    trac_obj = xlsx.shape[1]
    for n in range(1, 5):
      if n != int(xlsx.iloc[t, n-1]):
        # print(n, '->', xlsx.iloc[t, n-1])
        if xlsx.iloc[t, n-1] == 0:
          det_obj -= 1
          trac_obj -= 1
        else:
          trac_obj -= 1

    tpd = trac_obj / det_obj
    # print(tpd)
    score += tpd
    # print(score)
  print('TPD :', score/frame_size)


if __name__ == "__main__":
  path = './original_deepsort_labels.xlsx'
  print(path.split('/')[1], 'files operation..')
  evaluate_TPD(path)