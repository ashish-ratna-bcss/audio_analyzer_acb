"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-06-18
"""
from alembic import op
import sqlalchemy as sa

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "cases",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True)),
    )
    op.create_table(
        "files",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("case_id", sa.String(), sa.ForeignKey("cases.id"), index=True),
        sa.Column("original_filename", sa.String()),
        sa.Column("ext", sa.String()),
        sa.Column("source_sha256", sa.String(), nullable=True),
        sa.Column("status", sa.String()),
        sa.Column("created_at", sa.DateTime(timezone=True)),
    )
    op.create_table(
        "jobs",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("file_id", sa.String(), sa.ForeignKey("files.id"), index=True),
        sa.Column("case_id", sa.String(), sa.ForeignKey("cases.id"), index=True),
        sa.Column("status", sa.String(), index=True),
        sa.Column("stage", sa.String(), nullable=True),
        sa.Column("degraded_flags", sa.JSON()),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True)),
        sa.Column("updated_at", sa.DateTime(timezone=True)),
    )
    op.create_table(
        "segments",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("file_id", sa.String(), sa.ForeignKey("files.id"), index=True),
        sa.Column("start", sa.Float()),
        sa.Column("end", sa.Float()),
        sa.Column("speaker", sa.String(), nullable=True),
        sa.Column("text", sa.Text(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("source_pass", sa.String(), nullable=True),
        sa.Column("flagged", sa.Boolean()),
        sa.Column("review_status", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True)),
    )
    op.create_table(
        "reviews",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("segment_id", sa.String(), sa.ForeignKey("segments.id"), index=True),
        sa.Column("decision", sa.String()),
        sa.Column("text", sa.Text(), nullable=True),
        sa.Column("reviewer_id", sa.String()),
        sa.Column("created_at", sa.DateTime(timezone=True)),
    )
    op.create_table(
        "audit_entries",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("case_id", sa.String(), sa.ForeignKey("cases.id"), index=True),
        sa.Column("file_id", sa.String(), sa.ForeignKey("files.id"), nullable=True),
        sa.Column("stage", sa.String()),
        sa.Column("payload", sa.JSON()),
        sa.Column("prev_entry_hash", sa.String(), nullable=True),
        sa.Column("entry_hash", sa.String()),
        sa.Column("created_at", sa.DateTime(timezone=True)),
    )


def downgrade() -> None:
    for t in ["audit_entries", "reviews", "segments", "jobs", "files", "cases"]:
        op.drop_table(t)
