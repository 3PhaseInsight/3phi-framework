from sqlalchemy import select, update
from sqlalchemy.orm import Session

from threephi_framework.models.meta.workflow_state import WorkflowStateModel
from threephi_framework.resources.base import BaseResource


class WorkflowStateResource(BaseResource):
    def __init__(self, s: Session):
        super().__init__(s)

    def get(self, workflow: str) -> WorkflowStateModel | None:
        stmt = select(WorkflowStateModel).where(WorkflowStateModel.workflow == workflow)
        return self.s.execute(stmt).scalar_one_or_none()

    def get_or_create(self, workflow: str, description: str | None = None) -> WorkflowStateModel:
        obj = self.get(workflow)
        if obj is None:
            obj = WorkflowStateModel(workflow=workflow, completed=False, description=description)
            self.s.add(obj)
            self.s.flush()
        return obj

    def mark_completed(self, workflow: str) -> None:
        stmt = update(WorkflowStateModel).where(WorkflowStateModel.workflow == workflow).values(completed=True)
        self.s.execute(stmt)

    def mark_incomplete(self, workflow: str) -> None:
        stmt = update(WorkflowStateModel).where(WorkflowStateModel.workflow == workflow).values(completed=False)
        self.s.execute(stmt)

    def is_completed(self, workflow: str) -> bool:
        obj = self.get(workflow)
        return obj is not None and obj.completed
