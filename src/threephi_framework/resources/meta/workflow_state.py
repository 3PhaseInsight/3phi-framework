from sqlalchemy import insert, select, update
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
        stmt = (
            insert(WorkflowStateModel)
            .values(workflow=workflow, completed=False, description=description)
            .on_conflict_do_nothing(index_elements=[WorkflowStateModel.workflow])
        )
        self.s.execute(stmt)
        return self.get(workflow)

    def mark_completed(self, workflow: str) -> None:
        stmt = update(WorkflowStateModel).where(WorkflowStateModel.workflow == workflow).values(completed=True)
        self.s.execute(stmt)

    def mark_incomplete(self, workflow: str) -> None:
        stmt = update(WorkflowStateModel).where(WorkflowStateModel.workflow == workflow).values(completed=False)
        self.s.execute(stmt)

    def is_completed(self, workflow: str) -> bool:
        obj = self.get(workflow)
        return obj is not None and obj.completed
