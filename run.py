import json
import logging
import os
import pdb
import re
import time
from picamera2 import Picamera2
import numpy as np
from torch.autograd import Variable
from torch.nn import functional as F

from spatial import get_gesture_direction
from .mean import get_mean, get_std
from .model import generate_model
from .opts import parse_opts_offline as opt
from .spatial_transforms import *
from .utils import Queue

from .opts_run import args as opt


def weighting_func(x):
    return (1 / (1 + np.exp(-0.2 * (x - 9))))


regex = re.compile(r'(^module\.)')


def load_models(opt):
    opt.resume_path = opt.resume_path_det
    opt.pretrain_path = opt.pretrain_path_det
    opt.sample_duration = opt.sample_duration_det
    opt.model = opt.model_det
    opt.model_depth = opt.model_depth_det
    opt.width_mult = opt.width_mult_det
    opt.modality = opt.modality_det
    opt.resnet_shortcut = opt.resnet_shortcut_det
    opt.n_classes = opt.n_classes_det
    opt.n_finetune_classes = opt.n_finetune_classes_det

    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    opt.arch = '{}'.format(opt.model)
    opt.mean = get_mean(opt.norm_value)
    opt.std = get_std(opt.norm_value)

    print(opt)
    with open(os.path.join(opt.result_path, 'opts_det.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    torch.manual_seed(opt.manual_seed)

    detector, parameters = generate_model(opt)
    if not opt.no_cuda:
        detector = detector.cuda()
    if opt.resume_path:
        opt.resume_path = str(opt.resume_path)
        print('loading checkpoint {}'.format(opt.resume_path))
        if opt.no_cuda:
            checkpoint = torch.load(str(opt.resume_path), map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(str(opt.resume_path))

        # slice "module." from the model layer names
        # detector.load_state_dict([checkpoint['state_dict']])
        detector.load_state_dict({regex.sub('', k): v for k, v in checkpoint['state_dict'].items()})

    print('Model 1 \n', detector)
    pytorch_total_params = sum(p.numel() for p in detector.parameters() if
                               p.requires_grad)
    print("Total number of trainable parameters: ", pytorch_total_params)

    opt.resume_path = opt.resume_path_clf
    opt.pretrain_path = opt.pretrain_path_clf
    opt.sample_duration = opt.sample_duration_clf
    opt.model = opt.model_clf
    opt.model_depth = opt.model_depth_clf
    opt.width_mult = opt.width_mult_clf
    opt.modality = opt.modality_clf
    opt.resnet_shortcut = opt.resnet_shortcut_clf
    opt.n_classes = opt.n_classes_clf
    opt.n_finetune_classes = opt.n_finetune_classes_clf

    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    opt.arch = '{}'.format(opt.model)
    opt.mean = get_mean(opt.norm_value)
    opt.std = get_std(opt.norm_value)

    print(opt)
    with open(os.path.join(str(opt.result_path), 'opts_clf.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    torch.manual_seed(opt.manual_seed)
    classifier, parameters = generate_model(opt)
    if not opt.no_cuda:
        classifier = classifier.cuda()
    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        if opt.no_cuda:
            checkpoint = torch.load(str(opt.resume_path), map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(str(opt.resume_path))

        # slice "module." from the model layer names
        # classifier.load_state_dict(checkpoint['state_dict'])
        classifier.load_state_dict({regex.sub('', k): v for k, v in checkpoint['state_dict'].items()})

    print('Model 2 \n', classifier)
    pytorch_total_params = sum(p.numel() for p in classifier.parameters() if
                               p.requires_grad)
    print("Total number of trainable parameters: ", pytorch_total_params)

    return detector, classifier


detector, classifier = load_models(opt)

if opt.no_mean_norm and not opt.std_norm:
    norm_method = Normalize([0, 0, 0], [1, 1, 1])
elif not opt.std_norm:
    norm_method = Normalize(opt.mean, [1, 1, 1])
else:
    norm_method = Normalize(opt.mean, opt.std)

spatial_transform = Compose([
    Scale(112),
    CenterCrop(112),
    ToTensor(opt.norm_value), norm_method
])


def run(callback, capture_device: int = 0, playback: bool = False):
    num_frame = 0

    opt.sample_duration = max(opt.sample_duration_clf, opt.sample_duration_det)
    fps = ""
    clip = []
    active_index = 0
    passive_count = 0
    active = False
    prev_active = False
    finished_prediction = None
    pre_predict = False
    detector.eval()
    classifier.eval()
    cum_sum = np.zeros(opt.n_classes_clf, )
    clf_selected_queue = np.zeros(opt.n_classes_clf, )
    det_selected_queue = np.zeros(opt.n_classes_det, )
    myqueue_det = Queue(opt.det_queue_size, n_classes=opt.n_classes_det)
    myqueue_clf = Queue(opt.clf_queue_size, n_classes=opt.n_classes_clf)
    results = []
    prev_best1 = opt.n_classes_clf
    spatial_transform.randomize_parameters()

    if playback:
        cap = cv2.VideoCapture('/home/uw/test_ego.mp4')
    else:
        cap = Picamera2()
        cap.configure(cap.create_preview_configuration())
        cap.start()

    while True:
        t1 = time.time()
        if playback:
            _, frame = cap.read()
        else:
            frame = cap.capture_array()
        # cv2.imshow("Result", frame)
        if frame is None:
            logging.warning(f"Frame is {frame}")
            continue
        if num_frame == 0:
            cur_frame = cv2.resize(frame, (320, 240))
            cur_frame = Image.fromarray(cv2.cvtColor(cur_frame, cv2.COLOR_BGR2RGB))
            cur_frame = cur_frame.convert('RGB')
            for i in range(opt.sample_duration):
                clip.append(cur_frame)
            clip = [spatial_transform(img) for img in clip]
        clip.pop(0)
        _frame = cv2.resize(frame, (320, 240))
        _frame = Image.fromarray(cv2.cvtColor(_frame, cv2.COLOR_BGR2RGB))
        _frame = _frame.convert('RGB')
        _frame = spatial_transform(_frame)
        clip.append(_frame)
        im_dim = clip[0].size()[-2:]
        try:
            test_data = torch.cat(clip, 0).view((opt.sample_duration, -1) + im_dim).permute(1, 0, 2, 3)
        except Exception as e:
            pdb.set_trace()
            raise e
        inputs = torch.cat([test_data], 0).view(1, 3, opt.sample_duration, 112, 112)
        num_frame += 1

        with torch.no_grad():
            inputs = Variable(inputs)
            inputs_det = inputs[:, :, -opt.sample_duration_det:, :, :]
            outputs_det = detector(inputs_det)
            outputs_det = F.softmax(outputs_det, dim=1)
            outputs_det = outputs_det.cpu().numpy()[0].reshape(-1, )
            # enqueue the probabilities to the detector queue
            myqueue_det.enqueue(outputs_det.tolist())

            if opt.det_strategy == 'raw':
                det_selected_queue = outputs_det
            elif opt.det_strategy == 'median':
                det_selected_queue = myqueue_det.median
            elif opt.det_strategy == 'ma':
                det_selected_queue = myqueue_det.ma
            elif opt.det_strategy == 'ewma':
                det_selected_queue = myqueue_det.ewma
            prediction_det = np.argmax(det_selected_queue)

            prob_det = det_selected_queue[prediction_det]

            #### State of the detector is checked here as detector act as a switch for the classifier
            if prediction_det == 1:
                print('Classified as gesture')
                inputs_clf = inputs[:, :, :, :, :]
                inputs_clf = torch.Tensor(inputs_clf.numpy()[:, :, ::1, :, :])
                outputs_clf = classifier(inputs_clf)
                outputs_clf = F.softmax(outputs_clf, dim=1)
                outputs_clf = outputs_clf.cpu().numpy()[0].reshape(-1, )
                # Push the probabilities to queue
                myqueue_clf.enqueue(outputs_clf.tolist())
                passive_count = 0

                if opt.clf_strategy == 'raw':
                    clf_selected_queue = outputs_clf
                elif opt.clf_strategy == 'median':
                    clf_selected_queue = myqueue_clf.median
                elif opt.clf_strategy == 'ma':
                    clf_selected_queue = myqueue_clf.ma
                elif opt.clf_strategy == 'ewma':
                    clf_selected_queue = myqueue_clf.ewma

            else:
                outputs_clf = np.zeros(opt.n_classes_clf, )
                # Push the probabilities to queue
                myqueue_clf.enqueue(outputs_clf.tolist())
                passive_count += 1

        if passive_count >= opt.det_counter:
            active = False
        else:
            active = True

        # one of the following line need to be commented !!!!
        if active:
            active_index += 1
            cum_sum = ((cum_sum * (active_index - 1)) + (
                    weighting_func(active_index) * clf_selected_queue)) / active_index  # Weighted Aproach
            # cum_sum = ((cum_sum * (active_index-1)) + (1.0 * clf_selected_queue))/active_index #Not Weighting Aproach
            best2, best1 = tuple(cum_sum.argsort()[-2:][::1])
            if float(cum_sum[best1] - cum_sum[best2]) > opt.clf_threshold_pre:
                finished_prediction = True
                pre_predict = True
            print("Calling back with active gesture")
            callback({"gesture": best1, "direction": get_gesture_direction(best1), "confidence": cum_sum[best1].copy()})

        else:
            active_index = 0
        if active == False and prev_active == True:
            finished_prediction = True
        elif active == True and prev_active == False:
            finished_prediction = False

        if finished_prediction == True:
            # print(finished_prediction,pre_predict)
            best2, best1 = tuple(cum_sum.argsort()[-2:][::1])
            if cum_sum[best1] > opt.clf_threshold_final:
                if pre_predict == True:
                    if best1 != prev_best1:
                        if cum_sum[best1] > opt.clf_threshold_final:
                            results.append(((i * opt.stride_len) + opt.sample_duration_clf, best1))
                            print('Early Detected - class : {} with prob : {} at frame {}'.format(best1, cum_sum[best1],
                                                                                                  (
                                                                                                          i * opt.stride_len) + opt.sample_duration_clf))
                else:
                    if cum_sum[best1] > opt.clf_threshold_final:
                        if best1 == prev_best1:
                            if cum_sum[best1] > 5:
                                results.append(((i * opt.stride_len) + opt.sample_duration_clf, best1))
                                print('Late Detected - class : {} with prob : {} at frame {}'.format(best1,
                                                                                                     cum_sum[best1], (
                                                                                                             i * opt.stride_len) + opt.sample_duration_clf))
                        else:
                            results.append(((i * opt.stride_len) + opt.sample_duration_clf, best1))

                            print('Late Detected - class : {} with prob : {} at frame {}'.format(best1, cum_sum[best1],
                                                                                                 (
                                                                                                         i * opt.stride_len) + opt.sample_duration_clf))

                finished_prediction = False
                prev_best1 = best1

            cum_sum = np.zeros(opt.n_classes_clf, )

        if active == False and prev_active == True:
            pre_predict = False

        prev_active = active
        elapsedTime = time.time() - t1
        fps = "(Playback) {:.1f} FPS".format(1 / elapsedTime)

        if len(results) != 0:
            predicted = np.array(results)[:, 1]
            prev_best1 = -1
        else:
            predicted = []

        print('predicted classes: \t', predicted)

        cv2.putText(frame, fps, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38, 0, 255), 1, cv2.LINE_AA)
        cv2.imwrite('/tmp/output.png', frame)
        # cv2.imshow("Result", frame)

    cap.close()
    if playback:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run(lambda x: print(x), playback=True)
