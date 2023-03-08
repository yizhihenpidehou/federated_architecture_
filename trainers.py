# @Time   : 2022/3/8
# @Author : Bodian Ye

import numpy as np

import torch
from fedlab.core.client import ClientTrainer, SERIAL_TRAINER
from fedlab.utils import SerializationTool, Logger
from fedlab.utils.dataset import SubsetSampler
from torch.optim import Adam
from torch.utils.data import dataloader
import tqdm

from FMLP_Rec.utils import get_metric



class SerialTrainer(ClientTrainer):
    """Base class. Train multiple clients in sequence with a single process.

    Args:
        model (torch.nn.Module): Model used in this federation.
        client_num (int): Number of clients in current trainer.
        aggregator (Aggregators, callable, optional): Function to perform aggregation on a list of serialized model parameters.
        cuda (bool): Use GPUs or not. Default: ``True``.
        logger (Logger, optional): object of :class:`Logger`.
    """

    def __init__(self,
                 model,
                 client_num,
                 aggregator=None,
                 cuda=True,
                 logger=Logger()):
        super().__init__(model, cuda)
        self.client_num = client_num
        self.type = SERIAL_TRAINER  # represent serial trainer
        self.aggregator = aggregator
        self._LOGGER = logger

    def _train_alone(self, model_parameters, train_loader):
        """Train local model with :attr:`model_parameters` on :attr:`train_loader`.

        Args:
            model_parameters (torch.Tensor): Serialized model parameters of one model.
            train_loader (torch.utils.data.DataLoader): :class:`torch.utils.data.DataLoader` for this client.
        """
        raise NotImplementedError()

    def _get_dataloader(self, client_id):
        """Get :class:`DataLoader` for ``client_id``."""
        raise NotImplementedError()

    def train(self, model_parameters, id_list, aggregate=False):
        """Train local model with different dataset according to client id in ``id_list``.

        Args:
            model_parameters (torch.Tensor): Serialized model parameters.
            id_list (list[int]): Client id in this training serial.
            aggregate (bool): Whether to perform partial aggregation on this group of clients' local models at the end of each local training round.

        Note:
            Normally, aggregation is performed by server, while we provide :attr:`aggregate` option here to perform
            partial aggregation on current client group. This partial aggregation can reduce the aggregation workload
            of server.

        Returns:
            Serialized model parameters / list of model parameters.
        """
        param_list = []
        self._LOGGER.info(
            "Local training with client id list: {}".format(id_list))
        for idx in id_list:
            self._LOGGER.info(
                "Starting training procedure of client [{}]".format(idx))

            data_loader = self._get_dataloader(client_id=idx)
            self._train_alone(model_parameters=model_parameters,
                              train_loader=data_loader)
            param_list.append(self.model_parameters)

        if aggregate is True and self.aggregator is not None:
            # aggregate model parameters of this client group
            aggregated_parameters = self.aggregator(param_list)
            return aggregated_parameters
        else:
            return param_list


class SubsetSerialTrainer(SerialTrainer):
    """Train multiple clients in a single process.

    Customize :meth:`_get_dataloader` or :meth:`_train_alone` for specific algorithm design in clients.

    Args:
        model (torch.nn.Module): Model used in this federation.
        dataset (torch.utils.data.Dataset): Local dataset for this group of clients.
        data_slices (list[list]): subset of indices of dataset.
        aggregator (Aggregators, callable, optional): Function to perform aggregation on a list of model parameters.
        logger (Logger, optional): object of :class:`Logger`.
        cuda (bool): Use GPUs or not. Default: ``True``.
        args (dict, optional): Uncertain variables.

    .. note::
        ``len(data_slices) == client_num``, that is, each sub-index of :attr:`dataset` corresponds to a client's local dataset one-by-one.
    """

    def __init__(self,
                 model,
                 dataset,
                 data_slices,
                 aggregator=None,
                 logger=Logger(),
                 cuda=True,
                 args=None) -> None:

        super(SubsetSerialTrainer, self).__init__(model=model,
                                                  client_num=len(data_slices),
                                                  cuda=cuda,
                                                  aggregator=aggregator,
                                                  logger=logger)

        self.dataset = dataset
        self.data_slices = data_slices  # [0, client_num)
        self.args = args
        # self.args.log_file = "output/log.txt"
        self.optim = Adam(self.model.parameters(), lr=0.001, betas=(0.9,0.99), weight_decay=0.0)

    def _get_dataloader(self, client_id):
        """Return a training dataloader used in :meth:`train` for client with :attr:`id`

        Args:
            client_id (int): :attr:`client_id` of client to generate dataloader

        Note:
            :attr:`client_id` here is not equal to ``client_id`` in global FL setting. It is the index of client in current :class:`SerialTrainer`.

        Returns:
            :class:`DataLoader` for specific client's sub-dataset
        """
        batch_size = self.args["batch_size"]

        train_loader = torch.utils.data.DataLoader(
            self.dataset,
            sampler=SubsetSampler(indices=self.data_slices[client_id],
                                  shuffle=True),
            batch_size=batch_size)
        return train_loader

    def _train_alone(self, model_parameters, train_loader):
        """Single round of local training for one client.

        Note:
            Overwrite this method to customize the PyTorch training pipeline.

        Args:
            model_parameters (torch.Tensor): serialized model parameters.
            train_loader (torch.utils.data.DataLoader): :class:`torch.utils.data.DataLoader` for this client.
        """

        # for i, batch in rec_data_iter:
        #     # 0. batch_data will be sent into the device(GPU or CPU)
        #     batch = tuple(t.to(self.device) for t in batch)
        #     _, input_ids, answer, neg_answer = batch
        #     # Binary cross_entropy
        #     sequence_output = self.model(input_ids)
        #     # print("sequence_output:",sequence_output.shape)
        #
        #     loss = self.cross_entropy(sequence_output, answer, neg_answer)
        #
        #     self.optim.zero_grad()
        #     loss.backward()
        #     self.optim.step()
        #     rec_loss += loss.item()
        criterion = self.cross_entropy


        rec_loss = 0.0

        self._model.train()
        epochs  =self.args["epochs"]
        for _ in range(epochs):
            for train_data in train_loader:

                _, input_ids, answer, neg_answer = train_data

                # Binary cross_entropy
                sequence_output = self.model(input_ids)
                # print("sequence_output:",sequence_output.shape)

                loss = criterion(sequence_output, answer, neg_answer)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                rec_loss += loss.item()
        print("rec_loss:",rec_loss)
        return self.model_parameters

    def cross_entropy(self, seq_out, pos_ids, neg_ids):
        # [batch seq_len hidden_size]
        pos_emb = self.model.item_embeddings(pos_ids)
        neg_emb = self.model.item_embeddings(neg_ids)

        # [batch hidden_size]
        #pos = pos_emb.view(-1, pos_emb.size(2))
        #neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_out[:, -1, :] # [batch*seq_len hidden_size]
        pos_logits = torch.sum(pos_emb * seq_emb, -1) # [batch*seq_len]
        neg_logits = torch.sum(neg_emb * seq_emb, -1)
        #istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float() # [batch*seq_len]
        loss = torch.mean(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24)
        )# / torch.sum(istarget)

        return loss

    def predict_sample(self, seq_out, test_neg_sample):
        # [batch 100 hidden_size]
        test_item_emb = self.model.item_embeddings(test_neg_sample)
        # [batch hidden_size]
        test_logits = torch.bmm(test_item_emb, seq_out.unsqueeze(-1)).squeeze(-1)  # [B 100]
        return test_logits

    def get_metric(pred_list, topk=10):
        NDCG = 0.0
        HIT = 0.0
        MRR = 0.0
        # [batch] the answer's rank
        for rank in pred_list:
            MRR += 1.0 / (rank + 1.0)
            if rank < topk:
                NDCG += 1.0 / np.log2(rank + 2.0)
                HIT += 1.0
        return HIT / len(pred_list), NDCG / len(pred_list), MRR / len(pred_list)
    def get_sample_scores(self, epoch, pred_list):
        pred_list = (-pred_list).argsort().argsort()[:, 0]
        HIT_1, NDCG_1, MRR = get_metric(pred_list, 1)
        HIT_5, NDCG_5, MRR = get_metric(pred_list, 5)
        HIT_10, NDCG_10, MRR = get_metric(pred_list, 10)
        post_fix = {
            "Epoch": epoch,
            "HIT@1": '{:.4f}'.format(HIT_1), "NDCG@1": '{:.4f}'.format(NDCG_1),
            "HIT@5": '{:.4f}'.format(HIT_5), "NDCG@5": '{:.4f}'.format(NDCG_5),
            "HIT@10": '{:.4f}'.format(HIT_10), "NDCG@10": '{:.4f}'.format(NDCG_10),
            "MRR": '{:.4f}'.format(MRR),
        }
        print(post_fix)
        with open(self.args["log_file"], 'a') as f:
            f.write(str(post_fix) + '\n')
        return [HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR], str(post_fix)
    def c_evaluate(self, epoch, dataloader, full_sort=False, train=True):
        str_code = "test"
        rec_data_iter = tqdm.tqdm(enumerate(dataloader),
                                  desc="Recommendation EP_%s:%d" % (str_code, epoch),
                                  total=len(dataloader),
                                  bar_format="{l_bar}{r_bar}")

        for i, batch in rec_data_iter:
            # 0. batch_data will be sent into the device(GPU or cpu)
            batch = tuple(t for t in batch)
            user_ids, input_ids, answers, _, sample_negs = batch
            recommend_output = self.model(input_ids)
            # print("recommend_output:",recommend_output)
            test_neg_items = torch.cat((answers.unsqueeze(-1), sample_negs), -1)
            recommend_output = recommend_output[:, -1, :]

            test_logits = self.predict_sample(recommend_output, test_neg_items)
            test_logits = test_logits.cpu().detach().numpy().copy()
            if i == 0:
                pred_list = test_logits
            else:
                pred_list = np.append(pred_list, test_logits, axis=0)

        return self.get_sample_scores(epoch, pred_list)

