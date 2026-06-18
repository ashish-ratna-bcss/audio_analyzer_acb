"""segments: add detected_language column

Revision ID: 0004
Revises: 0003
"""
from alembic import op
import sqlalchemy as sa

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("segments", sa.Column("detected_language", sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column("segments", "detected_language")
