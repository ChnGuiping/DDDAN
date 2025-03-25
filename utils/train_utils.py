import logging
import os
import time
import warnings
import math
import torch
from torch import nn
from torch import optim
import models
import datasets
from loss.LM_Softmax import LM_Softmax
from loss.JDA_W import JDA_W



class train_utils(object):
    def __init__(self, args, save_dir):
        self.args = args
        self.save_dir = save_dir

    def setup(self):
        """
        Initialize the datasets, model, loss and optimizer:
        param args:
        return:
        """
        args = self.args

        # Consider the gpu or cpu condition
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
            assert args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('using {} cpu'.format(self.device_count))


        # Load the datasets
        Dataset = getattr(datasets, args.data_name)
        self.datasets = {}
        if isinstance(args.transfer_task[0], str):
           #print(args.transfer_task)
           args.transfer_task = eval("".join(args.transfer_task))
        self.datasets['source_train'], self.datasets['source_val'], self.datasets['target_train'], self.datasets['target_val'] = Dataset(args.data_dir, args.transfer_task, args.normlizetype).data_split(transfer_learning=True)


        self.dataloaders = {x: torch.utils.data.DataLoader(self.datasets[x], batch_size=args.batch_size,
                                                           shuffle=(True if x.split('_')[1] == 'train' else False),
                                                           num_workers=args.num_workers,
                                                           pin_memory=(True if self.device == 'cuda' else False),
                                                           drop_last=(True if args.last_batch and x.split('_')[1] == 'train' else False))
                            for x in ['source_train', 'source_val', 'target_train', 'target_val']}

        # Define the model
        self.model = getattr(models, args.model_name)(args.pretrained)
        if args.bottleneck:
            self.bottleneck_layer = nn.Sequential(nn.Linear(self.model.output_num(), args.bottleneck_num),nn.LeakyReLU(inplace=True))    # , nn.Dropout()
            self.classifier_layer = nn.Linear(args.bottleneck_num, Dataset.num_classes)
            self.model_all = nn.Sequential(self.model, self.bottleneck_layer, self.classifier_layer)
        else:
            self.classifier_layer = nn.Linear(self.model.output_num(), Dataset.num_classes)
            self.model_all = nn.Sequential(self.model, self.classifier_layer)


        # Define the learning parameters
        if args.bottleneck:
            parameter_list = [{"params": self.model.parameters(), "lr": args.lr},
                              {"params": self.bottleneck_layer.parameters(), "lr": args.lr},
                              {"params": self.classifier_layer.parameters(), "lr": args.lr}]
        else:
            parameter_list = [{"params": self.model.parameters(), "lr": args.lr},
                              {"params": self.classifier_layer.parameters(), "lr": args.lr}]


        # Define the optimizer
        if args.opt == 'sgd':
            self.optimizer = optim.SGD(parameter_list, lr=args.lr,
                                       momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.opt == 'adam':
            self.optimizer = optim.Adam(parameter_list, lr=args.lr,
                                        weight_decay=args.weight_decay)
        else:
            raise Exception("optimizer not implement")


        # Define the learning rate decay
        if args.lr_scheduler == 'step':
            steps = [int(step) for step in args.steps.split(',')]
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, steps, gamma=args.gamma)
        elif args.lr_scheduler == 'exp':
            self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, args.gamma)
        elif args.lr_scheduler == 'stepLR':
            steps = int(args.steps)
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, steps, args.gamma)
        elif args.lr_scheduler == 'fix':
            self.lr_scheduler = None
        else:
            raise Exception("lr schedule not implement")


        self.start_epoch = 0


        # Invert the model and define the loss
        self.model.to(self.device)
        if args.bottleneck:
            self.bottleneck_layer.to(self.device)
        self.classifier_layer.to(self.device)


        # Define the distance loss
        if args.distance_metric:
            if args.distance_loss == "JDA_W":
                self.softmax_layer = nn.Softmax(dim=1)
                self.softmax_layer = self.softmax_layer.to(self.device)
                self.distance_loss = JDA_W
            else:
                raise Exception("loss not implement")
        else:
            self.distance_loss = None

        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = LM_Softmax(Dataset.num_classes)


    def train(self):
        """
        Training process:
        return:
        """
        args = self.args
        Dataset = getattr(datasets, args.data_name)


        step = 0
        best_acc = 0.0

        for epoch in range(self.start_epoch, args.max_epoch):
            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-'*5)
            # Update the learning rate
            if self.lr_scheduler is not None:
                # self.lr_scheduler.step(epoch)
                logging.info('current lr: {}'.format(self.lr_scheduler.get_lr()))
            else:
                logging.info('current lr: {}'.format(args.lr))

            iter_target = iter(self.dataloaders['target_train'])
            len_target_loader = len(self.dataloaders['target_train'])
            # Each epoch has a training and val phase
            for phase in ['source_train', 'source_val', 'target_val']:
                # Define the temp variable
                epoch_start = time.time()
                epoch_acc = 0
                epoch_loss = 0.0
                epoch_length = 0

                # Set model to train mode or test mode
                if phase == 'source_train':
                    self.model.train()
                    if args.bottleneck:
                        self.bottleneck_layer.train()
                    self.classifier_layer.train()
                else:
                    self.model.eval()
                    if args.bottleneck:
                        self.bottleneck_layer.eval()
                    self.classifier_layer.eval()

                for batch_idx, (inputs, labels) in enumerate(self.dataloaders[phase]):
                    if phase != 'source_train' or epoch < args.middle_epoch:
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                    else:
                        source_inputs = inputs
                        # source_labels = labels
                        target_inputs, target_labels = next(iter_target)
                        inputs = torch.cat((source_inputs, target_inputs), dim=0)
                        # labels = torch.cat((source_labels, target_labels), dim=0)
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)

                    if (step + 1) % len_target_loader == 0:
                        iter_target = iter(self.dataloaders['target_train'])

                    with torch.set_grad_enabled(phase == 'source_train'):
                        # forward
                        features = self.model(inputs)
                        if args.bottleneck:
                            features = self.bottleneck_layer(features)
                        outputs = self.classifier_layer(features)
                        if phase != 'source_train' or epoch < args.middle_epoch:
                            logits = outputs
                            loss = self.criterion(logits, labels)
                        else:
                            logits = outputs.narrow(0, 0, labels.size(0))
                            classifier_loss = self.criterion(logits, labels)

                            logits2 = outputs.narrow(0, labels.size(0), inputs.size(0)-labels.size(0))
                            t_labels = logits2.argmax(dim=1)
                            labels1 = torch.cat((labels, t_labels), dim=0)
                            labels1 = labels1.to(self.device)


                            # Calculate the distance metric
                            if self.distance_loss is not None:
                                if args.distance_loss == 'JDA_W':
                                    softmax_out = self.softmax_layer(outputs)
                                    distance_loss = self.distance_loss([features.narrow(0, 0, args.batch_size),
                                                                        softmax_out.narrow(0, 0, args.batch_size)],
                                                                       [features.narrow(0, args.batch_size, inputs.size(0)-args.batch_size),
                                                                        softmax_out.narrow(0, args.batch_size, inputs.size(0)-args.batch_size)],
                                                                       labels1.narrow(0, 0, args.batch_size),
                                                                       labels1.narrow(0, args.batch_size,inputs.size(0) - args.batch_size),
                                                                       Dataset.num_classes
                                                                       )
                                else:
                                    raise Exception("loss not implement")

                            else:
                                distance_loss = 0

                            # Calculate the trade off parameter lam
                            if args.trade_off_distance == 'Cons':
                                lam_distance = args.lam_distance
                            elif args.trade_off_distance == 'Step':
                                lam_distance = 2 / (1 + math.exp(-10 * ((epoch-args.middle_epoch) / (args.max_epoch-args.middle_epoch)))) - 1
                            else:
                                raise Exception("trade_off_distance not implement")

                            loss = classifier_loss + lam_distance * distance_loss #+ lam_distance * adversarial_loss


                        pred = logits.argmax(dim=1)
                        correct = torch.eq(pred, labels).float().sum().item()
                        loss_temp = loss.item() * labels.size(0)
                        epoch_loss += loss_temp
                        epoch_acc += correct
                        epoch_length += labels.size(0)

                        # Calculate the training information
                        if phase == 'source_train':
                            # backward
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()

                            step += 1

                # Print the train and val information via each epoch
                epoch_loss = epoch_loss / epoch_length
                epoch_acc = epoch_acc / epoch_length

                logging.info('Epoch: {} {}-Loss: {:.4f} {}-Acc: {:.4f}, Cost {:.1f} sec'.format(
                    epoch, phase, epoch_loss, phase, epoch_acc, time.time() - epoch_start
                ))

                # save the model
                if phase == 'target_val':
                    # save the checkpoint for other learning
                    model_state_dic = self.model_all.state_dict()
                    # save the best model according to the val accuracy
                    if (epoch_acc > best_acc or epoch > args.max_epoch-2) and (epoch > args.middle_epoch-1):
                        best_acc = epoch_acc
                        logging.info("save best model epoch {}, acc {:.4f}".format(epoch, epoch_acc))
                        torch.save(model_state_dic,
                                   os.path.join(self.save_dir, '{}-{:.4f}-best_model.pth'.format(epoch, best_acc)))


            if self.lr_scheduler is not None:
                self.lr_scheduler.step()















