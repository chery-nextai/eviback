"""Evidence-Constrained Teacher components."""

from eviback.teacher.client import MockTeacherClient, OpenAICompatibleTeacherClient
from eviback.teacher.strategy import EvidenceConstrainedTeacher, TeacherDecision

__all__ = [
    "EvidenceConstrainedTeacher",
    "MockTeacherClient",
    "OpenAICompatibleTeacherClient",
    "TeacherDecision",
]