import numpy as np
import matplotlib.pyplot as plt
from tools import read_predicted_boxes, read_ground_truth_boxes


def calculate_iou(prediction_box, gt_box):
    """Calculate intersection over union of single predicted and ground truth box.

    Args:
        prediction_box (np.array of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (np.array of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]

        returns:
            float: value of the intersection of union for the two boxes.
    """

    # Checking if the boxes intersect
    if prediction_box[0] > gt_box[2]:
        return 0
    elif prediction_box[1] > gt_box[3]:
        return 0
    elif prediction_box[2] < gt_box[0]:
        return 0
    elif prediction_box[3] < gt_box[1]:
        return 0
    

    # Determine the intersection coordinates
    x_topLeft = max(prediction_box[0], gt_box[0])
    y_topLeft = max(prediction_box[1], gt_box[1])
    x_bottomRight = min(prediction_box[2], gt_box[2])
    y_bottomRight = min(prediction_box[3], gt_box[3])
    
    intersectionArea = (x_bottomRight - x_topLeft)*(y_bottomRight - y_topLeft)
    
    # Area of both the prediction and ground-truth rectangles
    boxPrediction = (prediction_box[2]-prediction_box[0])*(prediction_box[3]-prediction_box[1])
    boxGroundTruth = (gt_box[2]-gt_box[0])*(gt_box[3]-gt_box[1])

    # Compute the area of the union
    unionArea = boxPrediction + boxGroundTruth - intersectionArea

    iou = intersectionArea / unionArea
    assert iou >= 0 and iou <= 1
    return iou


def calculate_precision(num_tp, num_fp, num_fn):
    """ Calculates the precision for the given parameters.
        Returns 1 if num_tp + num_fp = 0

    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of precision
    """
    if num_tp + num_fp == 0:
        return 1.0
    return num_tp/(num_tp + num_fp)


def calculate_recall(num_tp, num_fp, num_fn):
    """ Calculates the recall for the given parameters.
        Returns 0 if num_tp + num_fn = 0
    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of recall
    """
    if num_tp + num_fn == 0:
        return 0.0
    return num_tp/(num_tp + num_fn)


def get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold):
    """Finds all possible matches for the predicted boxes to the ground truth boxes.
        No bounding box can have more than one match.

        Remember: Matching of bounding boxes should be done with decreasing IoU order!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns the matched boxes (in corresponding order):
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of box matches, 4].
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of box matches, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    """

    matches_iou = []
    # Iterate over all combinations of prediction and ground truth boxes to compute IoU values
    for i, pred_box in enumerate(prediction_boxes):
        for j, gt_box in enumerate(gt_boxes):
            iou = calculate_iou(pred_box, gt_box)
            if iou >= iou_threshold:
                # Store IoU value along with indices of the prediction and ground truth boxes
                matches_iou.append((iou, i, j))
    
    # Sort matches by IoU in descending order to prioritize higher IoU matches
    matches_iou.sort(reverse=True, key=lambda x: x[0])
    
    matched_gt_indices = set()
    matches_prediction = []
    matches_gt = []

    # Iterate over sorted IoU values and select matches
    for _, pred_index, gt_index in matches_iou:
        if gt_index not in matched_gt_indices:
            # Add the matched boxes to the lists
            matches_prediction.append(prediction_boxes[pred_index])
            matches_gt.append(gt_boxes[gt_index])
            # Mark this ground truth box as matched
            matched_gt_indices.add(gt_index)
    
    return np.array(matches_prediction), np.array(matches_gt)


def calculate_individual_image_result(prediction_boxes, gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes,
       calculates true positives, false positives and false negatives
       for a single image.
       NB: prediction_boxes and gt_boxes are not matched!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        dict: containing true positives, false positives, true negatives, false negatives
            {"true_pos": int, "false_pos": int, false_neg": int}
    """
    matched_prediction_indices, matched_gt_indices = get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold)
    tp = len(matched_prediction_indices)
    fp = len(prediction_boxes) - tp
    fn = len(gt_boxes) - len(matched_gt_indices)

    return {"true_pos": tp, "false_pos": fp, "false_neg": fn}
    

def calculate_precision_recall_all_images(
    all_prediction_boxes, all_gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates recall and precision over all images
       for a single image.
       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        tuple: (precision, recall). Both float.
    """
    total_true_pos = 0
    total_false_pos = 0
    total_false_neg = 0

    for prediction_boxes, gt_boxes in zip(all_prediction_boxes, all_gt_boxes):
        truth_values = calculate_individual_image_result(prediction_boxes, gt_boxes, iou_threshold)
        total_true_pos += truth_values["true_pos"]
        total_false_pos += truth_values["false_pos"]
        total_false_neg += truth_values["false_neg"]

    
    precision = calculate_precision(total_true_pos, total_false_pos, total_false_neg)
    recall = calculate_recall(total_true_pos, total_false_pos, total_false_neg)
    
    return (precision, recall)


def get_precision_recall_curve(
    all_prediction_boxes, all_gt_boxes, confidence_scores, iou_threshold
):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates the recall-precision curve over all images.
       for a single image.

       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        scores: (list of np.array of floats): each element in the list
            is a np.array containting the confidence score for each of the
            predicted bounding box. Shape: [number of predicted boxes]

            E.g: score[0][1] is the confidence score for a predicted bounding box 1 in image 0.
    Returns:
        precisions, recalls: two np.ndarray with same shape.
    """
    # Instead of going over every possible confidence score threshold to compute the PR
    # curve, we will use an approximation
    confidence_thresholds = np.linspace(0, 1, 500)
    precisions = [] 
    recalls = []
    # YOUR CODE HERE
    for threshold in confidence_thresholds:
        # Filter predictions based on the current threshold
        filtered_prediction_boxes = [boxes[scores >= threshold] for boxes, scores in zip(all_prediction_boxes, confidence_scores)]
        
        precision, recall = calculate_precision_recall_all_images(filtered_prediction_boxes, all_gt_boxes, iou_threshold)
        
        precisions.append(precision)
        recalls.append(recall)

    return np.array(precisions), np.array(recalls)


def plot_precision_recall_curve(precisions, recalls):
    """Plots the precision recall curve.
        Save the figure to precision_recall_curve.png:
        'plt.savefig("precision_recall_curve.png")'

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        None
    """
    plt.figure(figsize=(20, 20))
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.8, 1.0])
    plt.ylim([0.8, 1.0])
    plt.savefig("precision_recall_curve.png")


def calculate_mean_average_precision(precisions, recalls):
    """Given a precision recall curve, calculates the mean average precision.

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N

    Returns:
        float: mean average precision
    """
    # Define recall levels
    recall_levels = np.linspace(0, 1.0, 11)
    interpolated_precisions = 0

    for recall_level in recall_levels:
        # Find the highest precision for recall levels greater than or equal to the current level
        interpolated_precision = 0
        precisions_at_recall_level = precisions[recalls >= recall_level]
        if precisions_at_recall_level.size > 0:
            interpolated_precision = np.max(precisions_at_recall_level)
        interpolated_precisions += interpolated_precision

    # Calculate mean average precision
    average_precision = interpolated_precisions/11
    return average_precision


def mean_average_precision(ground_truth_boxes, predicted_boxes):
    """ Calculates the mean average precision over the given dataset
        with IoU threshold of 0.5

    Args:
        ground_truth_boxes: (dict)
        {
            "img_id1": (np.array of float). Shape [number of GT boxes, 4]
        }
        predicted_boxes: (dict)
        {
            "img_id1": {
                "boxes": (np.array of float). Shape: [number of pred boxes, 4],
                "scores": (np.array of float). Shape: [number of pred boxes]
            }
        }
    """
    # DO NOT EDIT THIS CODE
    all_gt_boxes = []
    all_prediction_boxes = []
    confidence_scores = []

    for image_id in ground_truth_boxes.keys():
        pred_boxes = predicted_boxes[image_id]["boxes"]
        scores = predicted_boxes[image_id]["scores"]

        all_gt_boxes.append(ground_truth_boxes[image_id])
        all_prediction_boxes.append(pred_boxes)
        confidence_scores.append(scores)

    precisions, recalls = get_precision_recall_curve(
        all_prediction_boxes, all_gt_boxes, confidence_scores, 0.5)
    plot_precision_recall_curve(precisions, recalls)
    mean_average_precision = calculate_mean_average_precision(precisions, recalls)
    print("Mean average precision: {:.4f}".format(mean_average_precision))


if __name__ == "__main__":
    ground_truth_boxes = read_ground_truth_boxes()
    predicted_boxes = read_predicted_boxes()
    mean_average_precision(ground_truth_boxes, predicted_boxes)


