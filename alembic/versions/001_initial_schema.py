"""Initial schema creation

Revision ID: 001
Revises: 
Create Date: 2025-08-16

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create initial database schema."""
    # Create UUID extension
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
    
    # Create recordings table
    op.create_table('recordings',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False, 
                  server_default=sa.text('uuid_generate_v4()')),
        sa.Column('file_path', sa.Text(), nullable=False),
        sa.Column('file_name', sa.Text(), nullable=False),
        sa.Column('sha256_hash', sa.String(64), nullable=True),
        sa.Column('xxh128_hash', sa.String(32), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), 
                  server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('sha256_hash'),
        sa.UniqueConstraint('xxh128_hash')
    )
    
    # Create metadata table
    op.create_table('metadata',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False,
                  server_default=sa.text('uuid_generate_v4()')),
        sa.Column('recording_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('key', sa.String(255), nullable=False),
        sa.Column('value', sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(['recording_id'], ['recordings.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes on metadata
    op.create_index('idx_metadata_recording_id', 'metadata', ['recording_id'])
    op.create_index('idx_metadata_key', 'metadata', ['key'])
    
    # Create tracklists table
    op.create_table('tracklists',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False,
                  server_default=sa.text('uuid_generate_v4()')),
        sa.Column('recording_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('source', sa.String(255), nullable=False),
        sa.Column('tracks', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('cue_file_path', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['recording_id'], ['recordings.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('recording_id')
    )


def downgrade() -> None:
    """Drop all tables and extensions."""
    op.drop_table('tracklists')
    op.drop_index('idx_metadata_key', table_name='metadata')
    op.drop_index('idx_metadata_recording_id', table_name='metadata')
    op.drop_table('metadata')
    op.drop_table('recordings')
    op.execute('DROP EXTENSION IF EXISTS "uuid-ossp"')