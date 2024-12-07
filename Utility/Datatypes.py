import math
import numpy as np

from dataclasses import dataclass
from statistics  import mean, median


@dataclass
class FlowPerformance:
    masked_epe: float
    epe : float
    px1 : float
    px3 : float
    px5 : float
    
    @classmethod
    def mean(cls, values: list["FlowPerformance"]) -> "FlowPerformance":
        return FlowPerformance(
            masked_epe  =sum([v.masked_epe for v in values]) / len(values),
            epe         =sum([v.epe        for v in values]) / len(values),
            px1         =sum([v.px1        for v in values]) / len(values),
            px3         =sum([v.px3        for v in values]) / len(values),
            px5         =sum([v.px5        for v in values]) / len(values),
        )


@dataclass
class FlowCovPerformance:
    masked_nll: float
    q25_nll: float
    q50_nll: float
    q75_nll: float
    
    uncertainty_avg_u: list[float]
    uncertainty_avg_v: list[float]
    uncertainty_bin  : tuple[float] = tuple(np.arange(0., 100. + 5, 5).tolist() + [float('inf')])
    
    @classmethod
    def mean(cls, values: list["FlowCovPerformance"]) -> "FlowCovPerformance":
        return FlowCovPerformance(
            masked_nll = mean([v.masked_nll for v in values]),
            q25_nll    = mean([v.q25_nll    for v in values]),
            q50_nll    = mean([v.q50_nll    for v in values]),
            q75_nll    = mean([v.q75_nll    for v in values]),
            uncertainty_avg_u = [mean([v.uncertainty_avg_u[idx] for v in values if not math.isnan(v.uncertainty_avg_u[idx])]) for idx in range(len(cls.uncertainty_bin) - 1)],
            uncertainty_avg_v = [mean([v.uncertainty_avg_v[idx] for v in values if not math.isnan(v.uncertainty_avg_v[idx])]) for idx in range(len(cls.uncertainty_bin) - 1)]
        )


@dataclass
class DepthPerformance:
    masked_err: float
    err_25 : float
    err_50 : float
    err_75 : float

    @classmethod
    def median(cls, values: list["DepthPerformance"]) -> "DepthPerformance":
        return DepthPerformance(
            masked_err= median([v.masked_err for v in values]),
            err_25    = median([v.err_25 for v in values]),
            err_50    = median([v.err_50 for v in values]),
            err_75    = median([v.err_75 for v in values]),
        )


@dataclass
class DepthCovPerformance:
    masked_nll: float
    q25_nll: float
    q50_nll: float
    q75_nll: float
    
    uncertainty_avg  : list[float]
    uncertainty_bin  : tuple[float] = tuple(np.arange(0., 100. + 5, 5).tolist() + [float('inf')])
    
    @classmethod
    def mean(cls, values: list["DepthCovPerformance"]) -> "DepthCovPerformance":
        return DepthCovPerformance(
            masked_nll = mean([v.masked_nll for v in values]),
            q25_nll    = mean([v.q25_nll    for v in values]),
            q50_nll    = mean([v.q50_nll    for v in values]),
            q75_nll    = mean([v.q75_nll    for v in values]),
            uncertainty_avg = [mean([v.uncertainty_avg[idx] for v in values if not math.isnan(v.uncertainty_avg[idx])]) for idx in range(len(cls.uncertainty_bin) - 1)],
        )
