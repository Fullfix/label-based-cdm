import pytorch_lightning as pl
import torch
import numpy as np
import abc
from typing import NamedTuple, Callable, Iterable
import torchmetrics
from pytorch_lightning.utilities.types import OptimizerLRScheduler
import pandas as pd
import networkx as nx


class CDMInputs(NamedTuple):
    student_idx: torch.LongTensor
    exercise_idx: torch.LongTensor


class NonNegClipper:
    """Clips negative values to 0 in all given models (excluding bias)"""

    def __init__(self, module_list: Iterable[torch.nn.Module]):
        self.module_list = module_list

    def apply(self):
        for module in self.module_list:
            if hasattr(module, 'weight'):
                with torch.no_grad():
                    module.weight.data = module.weight.clamp_min(0)


class CDMMixin(abc.ABC):
    """Abstract base class for all CDM models. Allows retrieval of mastery values and exercise parameters"""

    @abc.abstractmethod
    def get_mastery(self, student_idx: torch.LongTensor | None = None) -> np.ndarray:
        """Return mastery values for given student indices. If none are specified, return for all students"""

        pass

    @abc.abstractmethod
    def get_diff(self, exercise_idx: torch.LongTensor | None = None) -> np.ndarray:
        """Return difficulty for given exercise indices. If none are specified, return for all exercises"""

        pass

    @abc.abstractmethod
    def get_disc(self, exercise_idx: torch.LongTensor | None = None) -> np.ndarray:
        """Return discrimination for given exercise indices. If none are specified, return for all exercises"""

        pass


def degree_of_agreement(M: np.ndarray, R: np.ndarray, Q: np.ndarray):
    """
    Compute degree of agreement metric for all concepts

    :param M: mastery levels numpy array
    :param R: responses matrix numpy array
    :param Q: Q matrix numpy array
    :return: degree of agreement metric for each concept, numpy array of shape (K,)
    """

    K = Q.shape[1]
    S = M.shape[0]

    num = np.zeros((K,))
    denum = np.zeros((K,))
    for s1 in range(S):
        for s2 in range(S):
            if s1 == s2:
                continue
            mask = (M[s1] > M[s2])
            denum += mask

            num2 = (R[s1] > R[s2]) @ Q
            denum2 = (R[s1] != R[s2]) @ Q
            divres = np.divide(num2, denum2, out=np.zeros_like(num2).astype(float), where=denum2 != 0)
            num += mask * divres
    return num / denum


def degree_of_consistency(M: np.ndarray, R: np.ndarray, Q: np.ndarray):
    """
    Compute degree of consistency metric for all exercises

    :param M: mastery levels numpy array
    :param R: responses matrix numpy array
    :param Q: Q matrix numpy array
    :return: degree of consistency metric for each exercise, numpy array of shape (E,)
    """

    E = Q.shape[0]
    S = M.shape[0]

    num = np.zeros((E,))
    denum = np.zeros((E,))
    for s1 in range(S):
        for s2 in range(S):
            if s1 == s2:
                continue
            mask = (R[s1] > R[s2])
            denum += mask

            num2 = Q @ (M[s1] > M[s2])
            denum2 = Q @ (M[s1] != M[s2])
            divres = np.divide(num2, denum2, out=np.zeros_like(num2).astype(float), where=denum2 != 0)
            num += mask * divres
    return num / denum


class CDMDegreeOfAgreementMetric:
    """Degree of agreement metric for deep learning models"""

    def __init__(
            self,
            student_num: int,
            q_matrix: np.ndarray,
            get_mastery_fn: Callable[[], np.ndarray],
    ):
        self.student_num = student_num
        self.exercise_num = q_matrix.shape[0]
        self.concept_num = q_matrix.shape[1]
        self.q_matrix = q_matrix
        self.get_mastery_fn = get_mastery_fn
        self.responses = np.zeros((self.student_num, self.exercise_num))

    def update(self, inputs, labels) -> None:
        student_idx_new = inputs[0].cpu().detach().numpy()
        exercise_idx_new = inputs[1].cpu().detach().numpy()
        labels_new = labels.cpu().detach().numpy()

        self.responses[student_idx_new, exercise_idx_new] = labels_new

    def compute(self) -> float:
        mastery = self.get_mastery_fn()
        DOA = degree_of_agreement(mastery, self.responses, self.q_matrix)

        self.responses.fill(0)
        return np.mean(DOA)


class CDMDegreeOfConsistencyMetric:
    """Degree of consistency metric for deep learning models"""

    def __init__(
            self,
            student_num: int,
            q_matrix: np.ndarray,
            get_mastery_fn: Callable[[], np.ndarray],
    ):
        self.student_num = student_num
        self.exercise_num = q_matrix.shape[0]
        self.concept_num = q_matrix.shape[1]
        self.q_matrix = q_matrix
        self.get_mastery_fn = get_mastery_fn
        self.responses = np.zeros((self.student_num, self.exercise_num))

    def update(self, inputs, labels) -> None:
        student_idx_new = inputs[0].cpu().detach().numpy()
        exercise_idx_new = inputs[1].cpu().detach().numpy()
        labels_new = labels.cpu().detach().numpy()

        self.responses[student_idx_new, exercise_idx_new] = labels_new

    def compute(self) -> float:
        mastery = self.get_mastery_fn()
        DOC = degree_of_consistency(mastery, self.responses, self.q_matrix)

        self.responses.fill(0)
        return np.mean(DOC)


class MIRT(pl.LightningModule, CDMMixin):
    """
    Multidimensional IRT model
    :param cfg: omegaconf config, containing:
    - optimizer
    - - name: pytorch optimizer name
    - - params: optimizer parameters
    :param student_num: number of students
    :param exercise_num: number of exercises
    :param concept_num: number of concepts
    :param q_matrix: Q matrix, torch.FloatTensor
    """

    def __init__(
            self,
            cfg,
            student_num: int,
            exercise_num: int,
            concept_num: int,
            q_matrix: torch.FloatTensor
    ):
        super().__init__()
        self.cfg = cfg

        self.student_num = student_num
        self.exercise_num = exercise_num
        self.H_mast = torch.nn.Embedding(self.student_num, concept_num)
        self.H_disc = torch.nn.Embedding(self.exercise_num, concept_num)
        self.H_diff = torch.nn.Embedding(self.exercise_num, 1)
        self.register_buffer('q_matrix', q_matrix)

        self._criterion = torch.nn.BCELoss(reduction='mean')
        self._r2score = torchmetrics.R2Score()
        self._doa = CDMDegreeOfAgreementMetric(
            student_num=student_num,
            q_matrix=q_matrix.cpu().numpy(),
            get_mastery_fn=self.get_mastery
        )
        self._doc = CDMDegreeOfConsistencyMetric(
            student_num=student_num,
            q_matrix=q_matrix.cpu().numpy(),
            get_mastery_fn=self.get_mastery
        )

        for name, param in self.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_normal_(param)

    def forward(
            self,
            inputs: CDMInputs,
    ):
        student_idx, exercise_idx = inputs
        m = torch.sigmoid(self.H_mast(student_idx))
        h_disc = torch.sigmoid(self.H_disc(exercise_idx))
        h_diff = torch.sigmoid(self.H_diff(exercise_idx))
        if torch.max(m != m) or torch.max(h_disc != h_disc) or torch.max(h_diff != h_diff):
            raise ValueError('ValueError:theta,a,b may contains nan! The discrimination is too large.')
        concept_mask = self.q_matrix[exercise_idx]
        return torch.sigmoid((h_diff * concept_mask * (m - h_diff)).sum(dim=-1))

    def __share_step(self, batch):
        inputs, labels = batch

        preds = self.forward(inputs)
        loss = self._criterion(preds, labels)

        return loss, preds, labels

    def training_step(self, batch, batch_idx):
        loss, preds, labels = self.__share_step(batch)
        return {'loss': loss, 'preds': preds, 'labels': labels}

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self.__share_step(batch)
        self._r2score(preds, labels)
        self._doa.update(batch[0], labels)
        self._doc.update(batch[0], labels)
        return {'preds': preds, 'labels': labels}

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = eval(self.cfg.optimizer.name)(
            self.parameters(), **self.cfg.optimizer.params
        )
        return [optimizer]

    def on_validation_epoch_end(self) -> None:
        self.log(f'r2_epoch', self._r2score, prog_bar=True)
        self.log(f'DOA_epoch', self._doa.compute(), prog_bar=True)
        self.log(f'DOC_epoch', self._doc.compute(), prog_bar=True)

    def get_mastery(self, student_idx: torch.LongTensor | None = None) -> np.ndarray:
        student_idx = student_idx if student_idx is not None else torch.arange(self.student_num, device=self.device)
        return torch.sigmoid(self.H_mast(student_idx)).detach().cpu().numpy()

    def get_diff(self, exercise_idx: torch.LongTensor | None = None) -> np.ndarray:
        exercise_idx = exercise_idx if exercise_idx is not None else torch.arange(self.exercise_num, device=self.device)
        return torch.sigmoid(self.H_diff(exercise_idx)).detach().cpu().numpy()

    def get_disc(self, exercise_idx: torch.LongTensor | None = None) -> np.ndarray:
        exercise_idx = exercise_idx if exercise_idx is not None else torch.arange(self.exercise_num, device=self.device)
        return torch.sigmoid(self.H_disc(exercise_idx)).detach().cpu().numpy()


class NeuralCD(pl.LightningModule, CDMMixin):
    """
    NeuralCD model
    :param cfg: omegaconf config, containing:
    - optimizer
    - - name: pytorch optimizer name
    - - params: optimizer parameters
    - itf_layer1_dim: dimension of first interaction layer
    - itf_layer2_dim: dimension of second interaction layer
    :param student_num: number of students
    :param exercise_num: number of exercises
    :param concept_num: number of concepts
    :param q_matrix: Q matrix, torch.FloatTensor
    """


    def __init__(
            self,
            cfg,
            student_num: int,
            exercise_num: int,
            concept_num: int,
            q_matrix: torch.FloatTensor,
    ):
        super().__init__()

        self.cfg = cfg

        self.concept_num = concept_num
        self.exercise_num = exercise_num
        self.student_num = student_num
        self.prednet1_dim = cfg.itf_layer1_dim
        self.prednet2_dim = cfg.itf_layer2_dim
        self.register_buffer('q_matrix', q_matrix)
        self.H_mast = torch.nn.Embedding(self.student_num, self.concept_num)
        self.H_diff = torch.nn.Embedding(self.exercise_num, self.concept_num)
        self.H_disc = torch.nn.Embedding(self.exercise_num, 1)
        self.prednet_full1 = torch.nn.Linear(self.concept_num, self.prednet1_dim)
        self.prednet_full2 = torch.nn.Linear(self.prednet1_dim, self.prednet2_dim)
        self.relu = torch.nn.ReLU()
        self.prednet_full3 = torch.nn.Linear(self.prednet2_dim, 1)
        self.clipper = NonNegClipper([
            self.prednet_full1,
            self.prednet_full2,
            self.prednet_full3
        ])

        self._criterion = torch.nn.BCELoss(reduction='mean')
        self._r2score = torchmetrics.R2Score()
        self._doa = CDMDegreeOfAgreementMetric(
            student_num,
            q_matrix.cpu().numpy(),
            get_mastery_fn=self.get_mastery
        )
        self._doc = CDMDegreeOfConsistencyMetric(
            student_num,
            q_matrix.cpu().numpy(),
            get_mastery_fn=self.get_mastery
        )

        for name, param in self.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_normal_(param)

    def forward(self, inputs: CDMInputs):
        student_idx, exercise_idx = inputs

        m = torch.sigmoid(self.H_mast(student_idx))
        h_diff = torch.sigmoid(self.H_diff(exercise_idx))
        h_disc = torch.sigmoid(self.H_disc(exercise_idx)) * 10

        concept_mask = self.q_matrix[exercise_idx]
        x = h_disc * (m - h_diff) * concept_mask
        x = self.relu(self.prednet_full1(x))
        x = self.relu(self.prednet_full2(x))
        output = torch.sigmoid(self.prednet_full3(x))

        return output.squeeze(1)

    def __share_step(self, batch):
        inputs, labels = batch

        preds = self.forward(inputs)
        loss = self._criterion(preds, labels)

        return loss, preds, labels

    def training_step(self, batch, batch_idx):
        loss, preds, labels = self.__share_step(batch)
        return {'loss': loss, 'preds': preds, 'labels': labels}

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self.__share_step(batch)
        self._r2score(preds, labels)
        self._doa.update(batch[0], labels)
        self._doc.update(batch[0], labels)
        return {'preds': preds, 'labels': labels}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.clipper.apply()

    def on_validation_epoch_end(self) -> None:
        self.log(f'r2_epoch', self._r2score, prog_bar=True)
        self.log(f'DOA_epoch', self._doa.compute(), prog_bar=True)
        self.log(f'DOC_epoch', self._doc.compute(), prog_bar=True)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = eval(self.cfg.optimizer.name)(
            self.parameters(), **self.cfg.optimizer.params
        )
        return [optimizer]

    def get_mastery(self, student_idx: torch.LongTensor | None = None) -> np.ndarray:
        student_idx = student_idx if student_idx is not None else torch.arange(self.student_num, device=self.device)
        return torch.sigmoid(self.H_mast(student_idx)).detach().cpu().numpy()

    def get_diff(self, exercise_idx: torch.LongTensor | None = None) -> np.ndarray:
        exercise_idx = exercise_idx if exercise_idx is not None else torch.arange(self.exercise_num, device=self.device)
        return torch.sigmoid(self.H_diff(exercise_idx)).detach().cpu().numpy()

    def get_disc(self, exercise_idx: torch.LongTensor | None = None) -> np.ndarray:
        exercise_idx = exercise_idx if exercise_idx is not None else torch.arange(self.exercise_num, device=self.device)
        return torch.sigmoid(self.H_disc(exercise_idx)).detach().cpu().numpy()


class KaNCD(pl.LightningModule, CDMMixin):
    """
    Knowledge association NCD model, an extension of NeuralCD
    :param cfg: omegaconf config, containing:
    - optimizer
    - - name: pytorch optimizer name
    - - params: optimizer parameters
    - hidden_dim: dimension of student, exercises and concept embeddings
    - itf_layer1_dim: dimension of first interaction layer
    - itf_layer2_dim: dimension of second interaction layer
    - itf_type: interaction function type, one of ['mf', 'gmf', 'ncf1', 'ncf2']
    :param student_num: number of students
    :param exercise_num: number of exercises
    :param concept_num: number of concepts
    :param q_matrix: Q matrix, torch.FloatTensor
    """

    def __init__(
            self,
            cfg,
            student_num: int,
            exercise_num: int,
            concept_num: int,
            q_matrix: torch.FloatTensor,
    ):
        super().__init__()

        self.cfg = cfg

        self.concept_num = concept_num
        self.exercise_num = exercise_num
        self.student_num = student_num
        self.hidden_dim = cfg.hidden_dim
        self.itf_type = cfg.itf_type
        self.prednet_input_len = self.concept_num
        self.prednet_len1 = cfg.itf_layer1_dim
        self.prednet_len2 = cfg.itf_layer2_dim

        self.register_buffer('q_matrix', q_matrix)

        self.student_emb = torch.nn.Embedding(self.student_num, self.hidden_dim)
        self.exercise_emb = torch.nn.Embedding(self.exercise_num, self.hidden_dim)
        self.knowledge_emb = torch.nn.Parameter(torch.zeros(self.concept_num, self.hidden_dim))
        self.H_disc = torch.nn.Embedding(self.exercise_num, 1)
        self.prednet_full1 = torch.nn.Linear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = torch.nn.Dropout(p=0.5)
        self.prednet_full2 = torch.nn.Linear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = torch.nn.Dropout(p=0.5)
        self.prednet_full3 = torch.nn.Linear(self.prednet_len2, 1)

        if self.itf_type == 'gmf':
            self.k_diff_full = torch.nn.Linear(self.hidden_dim, 1)
            self.stat_full = torch.nn.Linear(self.hidden_dim, 1)
        elif self.itf_type == 'ncf1':
            self.k_diff_full = torch.nn.Linear(2 * self.hidden_dim, 1)
            self.stat_full = torch.nn.Linear(2 * self.hidden_dim, 1)
        elif self.itf_type == 'ncf2':
            self.k_diff_full1 = torch.nn.Linear(2 * self.hidden_dim, self.hidden_dim)
            self.k_diff_full2 = torch.nn.Linear(self.hidden_dim, 1)
            self.stat_full1 = torch.nn.Linear(2 * self.hidden_dim, self.hidden_dim)
            self.stat_full2 = torch.nn.Linear(self.hidden_dim, 1)
        self.clipper = NonNegClipper([
            self.prednet_full1,
            self.prednet_full2,
            self.prednet_full3
        ])

        self._criterion = torch.nn.BCELoss(reduction='mean')
        self._r2score = torchmetrics.R2Score()
        self._doa = CDMDegreeOfAgreementMetric(
            student_num,
            q_matrix.cpu().numpy(),
            get_mastery_fn=self.get_mastery
        )
        self._doc = CDMDegreeOfConsistencyMetric(
            student_num,
            q_matrix.cpu().numpy(),
            get_mastery_fn=self.get_mastery
        )

        for name, param in self.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_normal_(param)
        torch.nn.init.xavier_normal_(self.knowledge_emb)

    def forward(self, inputs: CDMInputs):
        student_idx, exercise_idx = inputs

        stu_emb = self.student_emb(student_idx)
        exer_emb = self.exercise_emb(exercise_idx)

        batch, dim = stu_emb.size()
        stu_emb = stu_emb.view(batch, 1, dim).repeat(1, self.concept_num, 1)
        knowledge_emb = self.knowledge_emb.repeat(batch, 1).view(batch, self.concept_num, -1)
        if self.itf_type == 'mf':  # simply inner product
            m = torch.sigmoid((stu_emb * knowledge_emb).sum(dim=-1, keepdim=False))  # batch, knowledge_n
        elif self.itf_type == 'gmf':
            m = torch.sigmoid(self.stat_full(stu_emb * knowledge_emb)).view(batch, -1)
        elif self.itf_type == 'ncf1':
            m = torch.sigmoid(self.stat_full(torch.cat((stu_emb, knowledge_emb), dim=-1))).view(batch, -1)
        elif self.itf_type == 'ncf2':
            m = torch.sigmoid(self.stat_full1(torch.cat((stu_emb, knowledge_emb), dim=-1)))
            m = torch.sigmoid(self.stat_full2(m)).view(batch, -1)
        else:
            raise ValueError(f'Invalid itf type: {self.itf_type}')

        batch, dim = exer_emb.size()
        exer_emb = exer_emb.view(batch, 1, dim).repeat(1, self.concept_num, 1)
        if self.itf_type == 'mf':
            h_diff = torch.sigmoid((exer_emb * knowledge_emb).sum(dim=-1, keepdim=False))  # batch, knowledge_n
        elif self.itf_type == 'gmf':
            h_diff = torch.sigmoid(self.k_diff_full(exer_emb * knowledge_emb)).view(batch, -1)
        elif self.itf_type == 'ncf1':
            h_diff = torch.sigmoid(self.k_diff_full(torch.cat((exer_emb, knowledge_emb), dim=-1))).view(batch, -1)
        elif self.itf_type == 'ncf2':
            h_diff = torch.sigmoid(self.k_diff_full1(torch.cat((exer_emb, knowledge_emb), dim=-1)))
            h_diff = torch.sigmoid(self.k_diff_full2(h_diff)).view(batch, -1)
        else:
            raise ValueError(f'Invalid itf type: {self.itf_type}')

        h_disc = torch.sigmoid(self.H_disc(exercise_idx))

        concept_mask = self.q_matrix[exercise_idx]
        x = h_disc * (m - h_diff) * concept_mask
        # f = input_x[input_knowledge_point == 1]
        x = self.drop_1(torch.tanh(self.prednet_full1(x)))
        x = self.drop_2(torch.tanh(self.prednet_full2(x)))
        output = torch.sigmoid(self.prednet_full3(x))

        return output.view(-1)

    def __share_step(self, batch):
        inputs, labels = batch

        preds = self.forward(inputs)
        loss = self._criterion(preds, labels)

        return loss, preds, labels

    def training_step(self, batch, batch_idx):
        loss, preds, labels = self.__share_step(batch)
        return {'loss': loss, 'preds': preds, 'labels': labels}

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self.__share_step(batch)
        self._r2score(preds, labels)
        self._doa.update(batch[0], labels)
        self._doc.update(batch[0], labels)
        return {'preds': preds, 'labels': labels}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.clipper.apply()

    def on_validation_epoch_end(self) -> None:
        self.log(f'r2_epoch', self._r2score, prog_bar=True)
        self.log(f'DOA_epoch', self._doa.compute(), prog_bar=True)
        self.log(f'DOC_epoch', self._doc.compute(), prog_bar=True)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = eval(self.cfg.optimizer.name)(
            self.parameters(), **self.cfg.optimizer.params
        )
        return [optimizer]

    def get_mastery(self, student_idx: torch.LongTensor | None = None) -> np.ndarray:
        student_idx = student_idx if student_idx is not None else torch.arange(self.student_num, device=self.device)
        stu_emb = self.student_emb(student_idx)

        batch, dim = stu_emb.size()
        stu_emb = stu_emb.view(batch, 1, dim).repeat(1, self.concept_num, 1)
        knowledge_emb = self.knowledge_emb.repeat(batch, 1).view(batch, self.concept_num, -1)
        if self.itf_type == 'mf':
            m = torch.sigmoid((stu_emb * knowledge_emb).sum(dim=-1, keepdim=False))  # batch, knowledge_n
        elif self.itf_type == 'gmf':
            m = torch.sigmoid(self.stat_full(stu_emb * knowledge_emb)).view(batch, -1)
        elif self.itf_type == 'ncf1':
            m = torch.sigmoid(self.stat_full(torch.cat((stu_emb, knowledge_emb), dim=-1))).view(batch, -1)
        elif self.itf_type == 'ncf2':
            m = torch.sigmoid(self.stat_full1(torch.cat((stu_emb, knowledge_emb), dim=-1)))
            m = torch.sigmoid(self.stat_full2(m)).view(batch, -1)
        else:
            raise ValueError(f'Invalid itf type: {self.itf_type}')

        return m.detach().cpu().numpy()

    def get_diff(self, exercise_idx: torch.LongTensor | None = None) -> np.ndarray:
        exercise_idx = exercise_idx if exercise_idx is not None else torch.arange(self.exercise_num, device=self.device)
        exer_emb = self.exercise_emb(exercise_idx)
        batch, dim = exer_emb.size()
        exer_emb = exer_emb.view(batch, 1, dim).repeat(1, self.concept_num, 1)
        knowledge_emb = self.knowledge_emb.repeat(batch, 1).view(batch, self.concept_num, -1)
        if self.itf_type == 'mf':
            h_diff = torch.sigmoid((exer_emb * knowledge_emb).sum(dim=-1, keepdim=False))  # batch, knowledge_n
        elif self.itf_type == 'gmf':
            h_diff = torch.sigmoid(self.k_diff_full(exer_emb * knowledge_emb)).view(batch, -1)
        elif self.itf_type == 'ncf1':
            h_diff = torch.sigmoid(self.k_diff_full(torch.cat((exer_emb, knowledge_emb), dim=-1))).view(batch, -1)
        elif self.itf_type == 'ncf2':
            h_diff = torch.sigmoid(self.k_diff_full1(torch.cat((exer_emb, knowledge_emb), dim=-1)))
            h_diff = torch.sigmoid(self.k_diff_full2(h_diff)).view(batch, -1)
        else:
            raise ValueError(f'Invalid itf type: {self.itf_type}')

        return h_diff.detach().cpu().numpy()

    def get_disc(self, exercise_idx: torch.LongTensor | None = None) -> np.ndarray:
        exercise_idx = exercise_idx if exercise_idx is not None else torch.arange(self.exercise_num, device=self.device)
        return torch.sigmoid(self.H_disc(exercise_idx)).detach().cpu().numpy()


class HierCDFLoss(torch.nn.Module):
    """Loss function for HierCDF, which ensures condi_p > condi_n"""

    def __init__(
            self,
            condi_p: torch.nn.Parameter,
            condi_n: torch.nn.Parameter,
            loss_fn: torch.nn.Module,
            factor=0.001
    ):
        super().__init__()
        self.condi_p = condi_p
        self.condi_n = condi_n
        self.factor = factor
        self.loss_fn = loss_fn

    def forward(self, student_idx, preds, labels):
        y_pred = preds
        y_target = labels
        return self.loss_fn(y_pred, y_target) + self.factor * torch.sum(
            torch.relu(self.condi_n[student_idx, :] - self.condi_p[student_idx, :]))


class HierCDF(pl.LightningModule, CDMMixin):
    """
    HierCDF model
    :param cfg: omegaconf config, containing:
    - optimizer
    - - name: pytorch optimizer name
    - - params: optimizer parameters
    - hidden_dim: dimension of student and exercise embeddings passed to interation function
    - itf_type: interaction function type, one of ['mirt', 'ncd']
    - loss_factor: loss factor for regularization term in HierCDFLoss
    :param student_num: number of students
    :param exercise_num: number of exercises
    :param concept_num: number of concepts
    :param q_matrix: Q matrix, torch.FloatTensor
    :param concept_graph: concept graph dataframe, consists of (from, to) rows representing edges
    """

    def __init__(
            self,
            cfg,
            student_num: int,
            exercise_num: int,
            concept_num: int,
            q_matrix: torch.FloatTensor,
            concept_graph: pd.DataFrame,  # (from, to) edges
    ):
        super().__init__()

        self.cfg = cfg

        self.student_num = student_num
        self.exercise_num = exercise_num
        self.concept_num = concept_num
        self.register_buffer('q_matrix', q_matrix)
        self.q_matrix.requires_grad = False
        hidden_dim = cfg.hidden_dim
        self.hidden_dim = hidden_dim
        self.concept_graph = concept_graph
        self.concept_edge = nx.DiGraph()
        for k in range(concept_num):
            self.concept_edge.add_node(k)
        for edge in concept_graph.values.tolist():
            self.concept_edge.add_edge(edge[0], edge[1])

        self.topo_order = list(nx.topological_sort(self.concept_edge))

        condi_p = torch.Tensor(student_num, concept_graph.shape[0])
        self.condi_p = torch.nn.Parameter(condi_p)

        condi_n = torch.Tensor(student_num, concept_graph.shape[0])
        self.condi_n = torch.nn.Parameter(condi_n)

        priori = torch.Tensor(student_num, concept_num)
        self.priori = torch.nn.Parameter(priori)

        self.H_diff = torch.nn.Embedding(exercise_num, concept_num)
        self.H_disc = torch.nn.Embedding(exercise_num, 1)

        self.user_contract = torch.nn.Linear(concept_num, hidden_dim)
        self.item_contract = torch.nn.Linear(concept_num, hidden_dim)

        self.cross_layer1 = torch.nn.Linear(hidden_dim, max(int(hidden_dim / 2), 1))
        self.cross_layer2 = torch.nn.Linear(max(int(hidden_dim / 2), 1), 1)

        self.itf_type = cfg.itf_type
        self.itf = {
            'mirt': self.mirt2pl,
            'ncd': self.ncd
        }[self.itf_type]
        self.clipper = NonNegClipper([
            self.user_contract,
            self.item_contract,
            self.cross_layer1,
            self.cross_layer2
        ])

        inner_loss_fn = torch.nn.BCELoss(reduction='mean')
        self._criterion = HierCDFLoss(
            condi_p=self.condi_p,
            condi_n=self.condi_n,
            loss_fn=inner_loss_fn,
            factor=cfg.loss_factor
        )
        self._r2score = torchmetrics.R2Score()
        self._doa = CDMDegreeOfAgreementMetric(
            student_num,
            q_matrix.cpu().numpy(),
            get_mastery_fn=self.get_mastery
        )
        self._doc = CDMDegreeOfConsistencyMetric(
            student_num,
            q_matrix.cpu().numpy(),
            get_mastery_fn=self.get_mastery
        )

        torch.nn.init.xavier_normal_(self.priori)
        torch.nn.init.xavier_normal_(self.condi_p)
        torch.nn.init.xavier_normal_(self.condi_n)
        for name, param in self.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_normal_(param)

    def mirt2pl(self, student_emb: torch.Tensor, exercise_emb: torch.Tensor, exercise_offset: torch.Tensor):
        return 1 / (1 + torch.exp(
            - torch.sum(torch.mul(student_emb, exercise_emb), axis=1).reshape(-1, 1) + exercise_offset))

    def ncd(self, student_emb: torch.Tensor, exercise_emb: torch.Tensor, exercise_offset: torch.Tensor):
        input_vec = (student_emb - exercise_emb) * exercise_offset
        x_vec = torch.sigmoid(self.cross_layer1(input_vec))
        x_vec = torch.sigmoid(self.cross_layer2(x_vec))
        return x_vec

    def get_posterior(self, student_idx: torch.LongTensor) -> torch.Tensor:
        n_batch = student_idx.shape[0]

        posterior = torch.rand(n_batch, self.concept_num, device=self.device)

        batch_priori = torch.sigmoid(self.priori[student_idx, :])
        batch_condi_p = torch.sigmoid(self.condi_p[student_idx, :])
        batch_condi_n = torch.sigmoid(self.condi_n[student_idx, :])

        for k in self.topo_order:
            # get predecessor list
            predecessors = list(self.concept_edge.predecessors(k))
            predecessors.sort()
            len_p = len(predecessors)

            # for each knowledge k, do:
            if len_p == 0:
                priori = batch_priori[:, k]
                posterior[:, k] = priori.reshape(-1)
                continue

            # format of masks
            fmt = '{0:0%db}' % (len_p)
            # number of parent master condition
            n_condi = 2 ** len_p

            # sigmoid to limit priori to (0,1)
            # priori = batch_priori[:,predecessors]
            priori = posterior[:, predecessors]

            # self.logger.write('priori:{}'.format(priori.requires_grad),'console')

            pred_idx = self.concept_graph[self.concept_graph['to'] == k].sort_values(by='from').index
            condi_p = torch.pow(batch_condi_p[:, pred_idx], 1 / len_p)
            condi_n = torch.pow(batch_condi_n[:, pred_idx], 1 / len_p)

            margin_p = condi_p * priori
            margin_n = condi_n * (1.0 - priori)

            posterior_k = torch.zeros((1, n_batch), device=self.device)

            for idx in range(n_condi):
                # for each parent mastery condition, do:
                mask = fmt.format(idx)
                mask = torch.tensor(np.array(list(mask)).astype(int), device=self.device)

                margin = mask * margin_p + (1 - mask) * margin_n
                margin = torch.prod(margin, dim=1).unsqueeze(dim=0)

                posterior_k = torch.cat([posterior_k, margin], dim=0)
            posterior_k = (torch.sum(posterior_k, dim=0)).squeeze()

            posterior[:, k] = posterior_k.reshape(-1)

        return posterior

    def get_condi_p(self, student_idx: torch.LongTensor) -> torch.Tensor:
        n_batch = student_idx.shape[0]
        result_tensor = torch.rand(n_batch, self.concept_num)
        batch_priori = torch.sigmoid(self.priori[student_idx, :])
        batch_condi_p = torch.sigmoid(self.condi_p[student_idx, :])

        # for k in range(self.n_know):
        for k in self.topo_order:
            # get predecessor list
            predecessors = list(self.concept_edge.predecessors(k))
            predecessors.sort()
            len_p = len(predecessors)
            if len_p == 0:
                priori = batch_priori[:, k]
                result_tensor[:, k] = priori.reshape(-1)
                continue
            pred_idx = self.concept_graph[self.concept_graph['to'] == k].sort_values(by='from').index
            condi_p = torch.pow(batch_condi_p[:, pred_idx], 1 / len_p)
            result_tensor[:, k] = torch.prod(condi_p, dim=1).reshape(-1)

        return result_tensor

    def get_condi_n(self, student_idx: torch.LongTensor) -> torch.Tensor:
        n_batch = student_idx.shape[0]
        result_tensor = torch.rand(n_batch, self.concept_num, device=self.device)
        batch_priori = torch.sigmoid(self.priori[student_idx, :])
        batch_condi_n = torch.sigmoid(self.condi_n[student_idx, :])

        # for k in range(self.n_know):
        for k in self.topo_order:
            # get predecessor list
            predecessors = list(self.concept_edge.predecessors(k))
            predecessors.sort()
            len_p = len(predecessors)
            if len_p == 0:
                priori = batch_priori[:, k]
                result_tensor[:, k] = priori.reshape(-1)
                continue
            pred_idx = self.concept_graph[self.concept_graph['to'] == k].sort_values(by='from').index
            condi_n = torch.pow(batch_condi_n[:, pred_idx], 1 / len_p)
            result_tensor[:, k] = torch.prod(condi_n, dim=1).reshape(-1)

        return result_tensor

    def forward(self, inputs: CDMInputs) -> torch.Tensor:
        student_idx, exercise_idx = inputs

        m = self.get_posterior(student_idx)
        h_exer = torch.sigmoid(self.H_diff(exercise_idx))
        h_offset = torch.sigmoid(self.H_disc(exercise_idx))

        concept_mask = self.q_matrix[exercise_idx]
        student_emb = torch.tanh(self.user_contract(m * concept_mask))
        exercise_emb = torch.sigmoid(self.item_contract(h_exer * concept_mask))

        output = self.itf(student_emb, exercise_emb, h_offset)

        return output.squeeze(1)

    def __share_step(self, batch):
        inputs, labels = batch

        preds = self.forward(inputs)
        loss = self._criterion(inputs[0], preds, labels)

        return loss, preds, labels

    def training_step(self, batch, batch_idx):
        loss, preds, labels = self.__share_step(batch)
        return {'loss': loss, 'preds': preds, 'labels': labels}

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self.__share_step(batch)
        self._r2score(preds, labels)
        self._doa.update(batch[0], labels)
        self._doc.update(batch[0], labels)
        return {'preds': preds, 'labels': labels}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.clipper.apply()

    def on_validation_epoch_end(self) -> None:
        self.log(f'r2_epoch', self._r2score, prog_bar=True)
        self.log(f'DOA_epoch', self._doa.compute(), prog_bar=True)
        self.log(f'DOC_epoch', self._doc.compute(), prog_bar=True)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = eval(self.cfg.optimizer.name)(
            self.parameters(), **self.cfg.optimizer.params
        )
        return [optimizer]

    def get_mastery(self, student_idx: torch.LongTensor | None = None) -> np.ndarray:
        student_idx = student_idx if student_idx is not None else torch.arange(self.student_num, device=self.device)
        return self.get_posterior(student_idx).detach().cpu().numpy()

    def get_diff(self, exercise_idx: torch.LongTensor | None = None) -> np.ndarray:
        exercise_idx = exercise_idx if exercise_idx is not None else torch.arange(self.exercise_num, device=self.device)
        return torch.sigmoid(self.H_diff(exercise_idx)).detach().cpu().numpy()

    def get_disc(self, exercise_idx: torch.LongTensor | None = None) -> np.ndarray:
        exercise_idx = exercise_idx if exercise_idx is not None else torch.arange(self.exercise_num, device=self.device)
        return torch.sigmoid(self.H_disc(exercise_idx)).detach().cpu().numpy()


class QCCDM(pl.LightningModule, CDMMixin):
    """
    QCCDM model
    :param cfg: omegaconf config, containing:
    - optimizer
    - - name: pytorch optimizer name
    - - params: optimizer parameters
    - layer_num: number of layers in interaction function
    - hidden_dim: dimension of student and exercise embeddings passed to interation function
    - nonlinear_fn_type: nonlinear function type, one of ['sigmoid', 'softplus', 'tanh'], is applied to embeddings
    - q_matrix_aug_enabled: whether to perform Q-matrix augmentation (bool)
    :param student_num: number of students
    :param exercise_num: number of exercises
    :param concept_num: number of concepts
    :param concept_graph_adj: concept graph adjacency matrix, torch.FloatTensor
    :param q_matrix: Q matrix, torch.FloatTensor
    """

    def __init__(
            self,
            cfg,
            student_num: int,
            exercise_num: int,
            concept_num: int,
            concept_graph_adj: torch.FloatTensor,
            q_matrix: torch.FloatTensor,
    ):
        super().__init__()

        self.cfg = cfg

        self.student_num = student_num
        self.exercise_num = exercise_num
        self.concept_num = concept_num
        self.nonlinear_fn_type = cfg.nonlinear_fn_type

        self.register_buffer('concept_graph_adj', concept_graph_adj)
        self.register_buffer('q_matrix', q_matrix)
        self.q_matrix.requires_grad = False

        self.q_matrix_aug = cfg.q_matrix_aug_enabled

        scm_layer_W = torch.randn(self.concept_num, self.concept_num)
        torch.nn.init.xavier_normal_(scm_layer_W)
        scm_layer_W = torch.sigmoid(scm_layer_W)
        self.scm_layer_W = torch.nn.Parameter(scm_layer_W)

        if self.q_matrix_aug:
            q_neural = torch.randn(self.exercise_num, self.concept_num)
            torch.nn.init.xavier_normal_(q_neural)
            q_neural = torch.sigmoid(q_neural)
            self.q_neural = torch.nn.Parameter(q_neural)

        self.latent_Zm_emb = torch.nn.Embedding(self.student_num, self.concept_num)
        self.latent_Zd_emb = torch.nn.Embedding(self.exercise_num, self.concept_num)
        self.H_disc = torch.nn.Embedding(self.exercise_num, 1)
        self.nonlinear_func = {
            'sigmoid': torch.nn.functional.sigmoid,
            'softplus': torch.nn.functional.softplus,
            'tanh': torch.nn.functional.tanh
        }[self.nonlinear_fn_type]

        layers = []
        clip_layers = []
        num_layers = cfg.layer_num
        hidden_dim = cfg.hidden_dim

        for i in range(num_layers):
            linear_layer = torch.nn.Linear(self.concept_num if i == 0 else hidden_dim // pow(2, i - 1),
                                           hidden_dim // pow(2, i))
            layers.append(linear_layer)
            clip_layers.append(linear_layer)
            layers.append(torch.nn.BatchNorm1d(hidden_dim // pow(2, i)))
            layers.append(torch.nn.Tanh())

        last_linear_layer = torch.nn.Linear(hidden_dim // pow(2, num_layers - 1), 1)
        layers.append(last_linear_layer)
        clip_layers.append(last_linear_layer)
        layers.append(torch.nn.Sigmoid())
        self.layers = torch.nn.Sequential(*layers)
        self.clipper = NonNegClipper(clip_layers)

        self._criterion = torch.nn.BCELoss(reduction='mean')
        self._r2score = torchmetrics.R2Score()
        self._doa = CDMDegreeOfAgreementMetric(
            student_num,
            q_matrix.cpu().numpy(),
            get_mastery_fn=self.get_mastery
        )
        self._doc = CDMDegreeOfConsistencyMetric(
            student_num,
            q_matrix.cpu().numpy(),
            get_mastery_fn=self.get_mastery
        )

        BatchNorm_names = ['layers.{}.weight'.format(3 * i + 1) for i in range(num_layers)]
        for index, (name, param) in enumerate(self.named_parameters()):
            if 'weight' in name:
                if name not in BatchNorm_names:
                    torch.nn.init.xavier_normal_(param)

    def forward(self, inputs: CDMInputs):
        student_idx, exercise_idx = inputs

        latent_zm = self.nonlinear_func(self.latent_Zm_emb(student_idx))
        latend_zd = self.nonlinear_func(self.latent_Zd_emb(exercise_idx))
        e_disc = torch.sigmoid(self.H_disc(exercise_idx))
        identity = torch.eye(self.concept_num).to(self.device)

        if self.nonlinear_fn_type != 'sigmoid':
            Mas = latent_zm @ (torch.inverse(identity - torch.mul(self.scm_layer_W, self.concept_graph_adj)))
            Mas = torch.sigmoid(self.nonlinear_func(Mas))
            Diff = latend_zd @ (torch.inverse(identity - torch.mul(self.scm_layer_W, self.concept_graph_adj)))
            Diff = torch.sigmoid(self.nonlinear_func(Diff))
            input_ability = Mas - Diff
        else:
            Mas = latent_zm @ (torch.inverse(identity - torch.mul(self.scm_layer_W, self.concept_graph_adj)))
            Mas = torch.sigmoid(Mas)
            Diff = latend_zd @ (torch.inverse(identity - torch.mul(self.scm_layer_W, self.concept_graph_adj)))
            Diff = torch.sigmoid(Diff)
            input_ability = Mas - Diff

        if self.q_matrix_aug:
            input_data = e_disc * input_ability * (self.q_neural * (1 - self.q_matrix) + self.q_matrix)[exercise_idx]
        else:
            concept_mask = self.q_matrix[exercise_idx]
            input_data = e_disc * input_ability * concept_mask

        return self.layers(input_data).view(-1)

    def __share_step(self, batch):
        inputs, labels = batch

        preds = self.forward(inputs)
        loss = self._criterion(preds, labels)

        return loss, preds, labels

    def training_step(self, batch, batch_idx):
        loss, preds, labels = self.__share_step(batch)
        return {'loss': loss, 'preds': preds, 'labels': labels}

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self.__share_step(batch)
        self._r2score(preds, labels)
        self._doa.update(batch[0], labels)
        self._doc.update(batch[0], labels)
        return {'preds': preds, 'labels': labels}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.clipper.apply()

    def on_validation_epoch_end(self) -> None:
        self.log(f'r2_epoch', self._r2score, prog_bar=True)
        self.log(f'DOA_epoch', self._doa.compute(), prog_bar=True)
        self.log(f'DOC_epoch', self._doc.compute(), prog_bar=True)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = eval(self.cfg.optimizer.name)(
            self.parameters(), **self.cfg.optimizer.params
        )
        return [optimizer]

    def get_mastery(self, student_idx: torch.LongTensor | None = None) -> np.ndarray:
        student_idx = student_idx if student_idx is not None else torch.arange(self.student_num)
        identity = torch.eye(self.concept_num)
        if self.nonlinear_fn_type != 'sigmoid':
            return torch.sigmoid(
                self.nonlinear_func(self.nonlinear_func(self.latent_Zm_emb.weight.detach().cpu()) @ torch.linalg.inv(
                    identity - self.scm_layer_W.data.detach().cpu() * self.concept_graph_adj.detach().cpu())))[
                student_idx].numpy()
        else:
            return torch.sigmoid(
                self.nonlinear_func(self.latent_Zm_emb.weight.detach().cpu()) @ torch.linalg.inv(
                    identity - self.scm_layer_W.data.detach().cpu() * self.concept_graph_adj.detach().cpu()))[
                student_idx].numpy()

    def get_diff(self, exercise_idx: torch.LongTensor | None = None) -> np.ndarray:
        exercise_idx = exercise_idx if exercise_idx is not None else torch.arange(self.exercise_num, device=self.device)

        identity = torch.eye(self.concept_num).to(self.device)
        latend_zd = self.nonlinear_func(self.latent_Zd_emb(exercise_idx))
        Diff = latend_zd @ (torch.inverse(identity - torch.mul(self.scm_layer_W, self.concept_graph_adj)))
        Diff = torch.sigmoid(Diff)
        return Diff.detach().cpu().numpy()

    def get_disc(self, exercise_idx: torch.LongTensor | None = None) -> np.ndarray:
        exercise_idx = exercise_idx if exercise_idx is not None else torch.arange(self.exercise_num, device=self.device)
        return torch.sigmoid(self.H_disc(exercise_idx)).detach().cpu().numpy()