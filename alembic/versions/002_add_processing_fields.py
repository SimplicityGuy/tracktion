"""Add processing fields to recordings table

Revision ID: 002
Revises: 001
Create Date: 2025-08-16

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '002'
down_revision = '001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add processing fields to recordings table."""
    # Add new columns
    op.add_column('recordings', sa.Column('file_hash', sa.String(64), nullable=True))
    op.add_column('recordings', sa.Column('file_size', sa.Integer(), nullable=True))
    op.add_column('recordings', sa.Column('processing_status', sa.String(50), nullable=True, server_default='pending'))
    op.add_column('recordings', sa.Column('processing_error', sa.Text(), nullable=True))
    op.add_column('recordings', sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    """Remove processing fields from recordings table."""
    op.drop_column('recordings', 'updated_at')
    op.drop_column('recordings', 'processing_error')
    op.drop_column('recordings', 'processing_status')
    op.drop_column('recordings', 'file_size')
    op.drop_column('recordings', 'file_hash')