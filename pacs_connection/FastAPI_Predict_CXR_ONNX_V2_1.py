def plot_bbox_from_df(df_bbox, image_path):

    ### df_bbox คือ json ที่ได้มาจาก database
    ### image_path คือ path ของ image

    import cv2
    import pydicom
    import numpy as np
    import pandas as pd
    import random

    all_bbox = pd.DataFrame(df_bbox['data'], columns=['label', 'tool', 'data'])

    #generate random class-color mapping
    all_class = all_bbox["label"].unique()
    cmap = dict()
    for lb in all_class:
        while True:
            b = random.randint(0,255)
            g = random.randint(0,255)
            r = random.randint(0,255)
            if (b,g,r) not in cmap.values():
                cmap[lb] = (b,g,r)
                break
    all_bbox["color"] = all_bbox["label"].map(cmap)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1.5
    thickness = 6
    isClosed = True

    ### ส่วนที่เรียก image
    #import dicom image array
    if image_path != None:
        dicom = pydicom.dcmread(("img_path"))
        inputImage = dicom.pixel_array

        # depending on this value, X-ray may look inverted - fix that:
        if dicom.PhotometricInterpretation == "MONOCHROME1":
            inputImage = np.amax(inputImage) - inputImage
            inputImage = np.stack([inputImage, inputImage, inputImage])
            inputImage = inputImage.astype('float32')
            inputImage = inputImage - inputImage.min()
            inputImage = inputImage / inputImage.max()
            inputImage = inputImage.transpose(1, 2, 0)
            inputImage = (inputImage*255).astype(int)
            # https://github.com/opencv/opencv/issues/14866
            inputImage = cv2.UMat(inputImage).get()
    else:
        inputImage = np.zeros([3000, 3000, 3])
        inputImage = inputImage.astype(int)
        inputImage = cv2.UMat(inputImage).get()

    all_ratio = []
    for index, row in all_bbox.iterrows():

        if row['tool'] == 'rectangleRoi':
            pts = row["data"]["handles"]
            inputImage = cv2.rectangle(inputImage, (int(pts["start"]["x"]), int(pts["start"]["y"])), (int(
                pts["end"]["x"]), int(pts["end"]["y"])), row["color"], thickness)
            inputImage = cv2.putText(inputImage,  row["label"], (int(min(pts["start"]["x"], pts["end"]["x"])), int(
                min(pts["start"]["y"], pts["end"]["y"]))), font, fontScale, row["color"], thickness, cv2.LINE_AA)

        if row['tool'] == 'freehand':
            pts = np.array([[cdn["x"], cdn["y"]]
                           for cdn in row["data"]["handles"]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            #choose min x,y for text origin
            text_org = np.amin(pts, axis=0)
            inputImage = cv2.polylines(inputImage, [pts], isClosed, row["color"], thickness)
            inputImage = cv2.putText(inputImage,  row["label"], tuple(
                text_org[0]), font, fontScale, row["color"], thickness, cv2.LINE_AA)

        if row['tool'] == 'length':
            pts = row["data"]["handles"]
            #choose left point for text origin
            text_org = "start"
            if pts["start"]["x"] > pts["end"]["x"]:
                text_org = "end"
            inputImage = cv2.line(inputImage, (int(pts["start"]["x"]), int(pts["start"]["y"])), (int(
                pts["end"]["x"]), int(pts["end"]["y"])), row["color"], thickness)
            inputImage = cv2.putText(inputImage,  row["label"], (int(pts[text_org]["x"]), int(
                pts[text_org]["y"])), font, fontScale, row["color"], thickness, cv2.LINE_AA)

        if row['tool'] == 'ratio':
            all_ratio.append((row["label"],row["data"]["ratio"],row["color"]))
    
    #write all ratios on top-left
    for i in range(len(all_ratio)):
        inputImage = cv2.putText(inputImage,  "{} ({})".format(all_ratio[i][0],all_ratio[i][1]), (10,50+60*i), font, fontScale, all_ratio[i][2], thickness, cv2.LINE_AA)

        #cv2.imshow(inputImage)
        #cv2.imwrite("/path/to/save.png", inputImage)
    return inputImage
        
