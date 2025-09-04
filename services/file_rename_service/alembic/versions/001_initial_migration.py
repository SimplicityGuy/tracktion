"""Initial migration for file rename service

Revision ID: 001
Revises:
Create Date: 2024-01-29

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op  # type: ignore[attr-defined]  # Alembic adds attributes at runtime

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Create pattern_categories table
    op.create_table(
        "pattern_categories",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("parent_category_id", sa.Integer(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(
            ["parent_category_id"],
            ["pattern_categories.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )
    op.create_index("idx_category_name", "pattern_categories", ["name"], unique=False)
    op.create_index(
        "idx_category_parent",
        "pattern_categories",
        ["parent_category_id"],
        unique=False,
    )

    # Create patterns table
    op.create_table(
        "patterns",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("pattern_type", sa.String(length=50), nullable=False),
        sa.Column("pattern_value", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("category", sa.String(length=100), nullable=True),
        sa.Column("frequency", sa.Integer(), nullable=True),
        sa.Column("confidence_score", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_pattern_confidence", "patterns", ["confidence_score"], unique=False)
    op.create_index(
        "idx_pattern_type_category",
        "patterns",
        ["pattern_type", "category"],
        unique=False,
    )
    op.create_index(op.f("ix_patterns_id"), "patterns", ["id"], unique=False)

    # Create ml_models table
    op.create_table(
        "ml_models",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("model_name", sa.String(length=255), nullable=False),
        sa.Column("model_version", sa.String(length=50), nullable=False),
        sa.Column("model_type", sa.String(length=100), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=True),
        sa.Column("accuracy", sa.Float(), nullable=True),
        sa.Column("precision", sa.Float(), nullable=True),
        sa.Column("recall", sa.Float(), nullable=True),
        sa.Column("f1_score", sa.Float(), nullable=True),
        sa.Column("training_samples", sa.Integer(), nullable=True),
        sa.Column("training_duration_seconds", sa.Float(), nullable=True),
        sa.Column("model_path", sa.Text(), nullable=True),
        sa.Column("parameters", sa.JSON(), nullable=True),
        sa.Column("training_metadata", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.Column("deployed_at", sa.DateTime(), nullable=True),
        sa.Column("deprecated_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("model_name"),
        sa.UniqueConstraint("model_name", "model_version", name="uq_model_name_version"),
    )
    op.create_index("idx_model_status", "ml_models", ["status"], unique=False)
    op.create_index("idx_model_type", "ml_models", ["model_type"], unique=False)
    op.create_index(op.f("ix_ml_models_id"), "ml_models", ["id"], unique=False)

    # Create rename_history table
    op.create_table(
        "rename_history",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("original_name", sa.Text(), nullable=False),
        sa.Column("proposed_name", sa.Text(), nullable=False),
        sa.Column("final_name", sa.Text(), nullable=True),
        sa.Column("file_path", sa.Text(), nullable=True),
        sa.Column("file_extension", sa.String(length=50), nullable=True),
        sa.Column("file_size_bytes", sa.Integer(), nullable=True),
        sa.Column("pattern_id", sa.Integer(), nullable=True),
        sa.Column("ml_model_id", sa.Integer(), nullable=True),
        sa.Column("confidence_score", sa.Float(), nullable=True),
        sa.Column("was_accepted", sa.Boolean(), nullable=True),
        sa.Column("user_feedback", sa.Text(), nullable=True),
        sa.Column("feedback_rating", sa.Integer(), nullable=True),
        sa.Column("processing_time_ms", sa.Float(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(
            ["ml_model_id"],
            ["ml_models.id"],
        ),
        sa.ForeignKeyConstraint(
            ["pattern_id"],
            ["patterns.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_rename_accepted", "rename_history", ["was_accepted"], unique=False)
    op.create_index("idx_rename_confidence", "rename_history", ["confidence_score"], unique=False)
    op.create_index("idx_rename_created", "rename_history", ["created_at"], unique=False)
    op.create_index(op.f("ix_rename_history_id"), "rename_history", ["id"], unique=False)

    # Create user_feedback table
    op.create_table(
        "user_feedback",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("rename_history_id", sa.Integer(), nullable=True),
        sa.Column("feedback_type", sa.String(length=50), nullable=True),
        sa.Column("corrected_name", sa.Text(), nullable=True),
        sa.Column("rating", sa.Integer(), nullable=True),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column("is_helpful", sa.Boolean(), nullable=True),
        sa.Column("user_id", sa.String(length=255), nullable=True),
        sa.Column("session_id", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(
            ["rename_history_id"],
            ["rename_history.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("rename_history_id"),
    )
    op.create_index("idx_feedback_created", "user_feedback", ["created_at"], unique=False)
    op.create_index("idx_feedback_rating", "user_feedback", ["rating"], unique=False)
    op.create_index("idx_feedback_type", "user_feedback", ["feedback_type"], unique=False)
    op.create_index(op.f("ix_user_feedback_id"), "user_feedback", ["id"], unique=False)


def downgrade() -> None:
    # Drop all tables in reverse order
    op.drop_index(op.f("ix_user_feedback_id"), table_name="user_feedback")
    op.drop_index("idx_feedback_type", table_name="user_feedback")
    op.drop_index("idx_feedback_rating", table_name="user_feedback")
    op.drop_index("idx_feedback_created", table_name="user_feedback")
    op.drop_table("user_feedback")

    op.drop_index(op.f("ix_rename_history_id"), table_name="rename_history")
    op.drop_index("idx_rename_created", table_name="rename_history")
    op.drop_index("idx_rename_confidence", table_name="rename_history")
    op.drop_index("idx_rename_accepted", table_name="rename_history")
    op.drop_table("rename_history")

    op.drop_index(op.f("ix_ml_models_id"), table_name="ml_models")
    op.drop_index("idx_model_type", table_name="ml_models")
    op.drop_index("idx_model_status", table_name="ml_models")
    op.drop_table("ml_models")

    op.drop_index(op.f("ix_patterns_id"), table_name="patterns")
    op.drop_index("idx_pattern_type_category", table_name="patterns")
    op.drop_index("idx_pattern_confidence", table_name="patterns")
    op.drop_table("patterns")

    op.drop_index("idx_category_parent", table_name="pattern_categories")
    op.drop_index("idx_category_name", table_name="pattern_categories")
    op.drop_table("pattern_categories")
