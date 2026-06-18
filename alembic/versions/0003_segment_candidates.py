"""segment candidates + clips

Revision ID: 0003
Revises: 0002
"""
from alembic import op
import sqlalchemy as sa

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("segments", sa.Column("candidates", sa.JSON(), nullable=True))
    op.add_column("segments", sa.Column("clip_original", sa.String(), nullable=True))
    op.add_column("segments", sa.Column("clip_enhanced", sa.String(), nullable=True))


def downgrade() -> None:
    for c in ["candidates", "clip_original", "clip_enhanced"]:
        op.drop_column("segments", c)
