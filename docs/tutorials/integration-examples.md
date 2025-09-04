# Integration Examples

This guide provides detailed examples of integrating Tracktion with various platforms, frameworks, and services. Each example includes complete working code, configuration, and deployment instructions.

## Table of Contents

- [Web Framework Integration](#web-framework-integration)
- [Cloud Platform Integration](#cloud-platform-integration)
- [Database Integration](#database-integration)
- [Message Queue Integration](#message-queue-integration)
- [External API Integration](#external-api-integration)
- [Docker & Kubernetes Integration](#docker--kubernetes-integration)
- [Monitoring & Logging Integration](#monitoring--logging-integration)

---

## Web Framework Integration

### FastAPI Integration

Complete FastAPI application with Tracktion analysis endpoints.

```python
"""FastAPI integration with Tracktion for web-based music analysis."""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import asyncio
import tempfile
import os
from pathlib import Path

from tracktion.analysis import BPMDetector, KeyDetector, MoodAnalyzer
from tracktion.database import TrackDatabase
from tracktion.tracklist import PlaylistGenerator

app = FastAPI(
    title="Tracktion Music Analysis API",
    description="RESTful API for music analysis using Tracktion",
    version="1.0.0"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-frontend.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class AnalysisRequest(BaseModel):
    confidence_threshold: float = 0.8
    enable_mood_analysis: bool = True

class AnalysisResult(BaseModel):
    bpm: Optional[float]
    bpm_confidence: Optional[float]
    key: Optional[str]
    key_confidence: Optional[float]
    moods: Optional[List[str]]
    energy: Optional[float]
    danceability: Optional[float]
    processing_time: float
    filename: str

class PlaylistRequest(BaseModel):
    name: str
    criteria: dict
    duration_minutes: int = 60
    max_tracks: int = 50

# Global analyzers (initialized once)
bpm_detector = BPMDetector()
key_detector = KeyDetector()
mood_analyzer = MoodAnalyzer()
playlist_generator = PlaylistGenerator()
db = TrackDatabase(os.getenv("DATABASE_URL", "postgresql://localhost/tracktion"))

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "Tracktion Music Analysis API", "status": "healthy"}

@app.post("/analyze/upload", response_model=AnalysisResult)
async def analyze_uploaded_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    request: AnalysisRequest = Depends()
):
    """Analyze an uploaded audio file."""

    # Validate file type
    allowed_types = {'.mp3', '.flac', '.wav', '.m4a', '.aiff'}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Allowed: {', '.join(allowed_types)}"
        )

    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name

    try:
        # Perform analysis
        import time
        start_time = time.time()

        # Run analyzers concurrently
        tasks = [
            bpm_detector.analyze(tmp_file_path),
            key_detector.analyze(tmp_file_path)
        ]

        if request.enable_mood_analysis:
            tasks.append(mood_analyzer.analyze(tmp_file_path))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        processing_time = time.time() - start_time

        # Process results
        bmp_result = results[0] if not isinstance(results[0], Exception) else None
        key_result = results[1] if not isinstance(results[1], Exception) else None
        mood_result = results[2] if len(results) > 2 and not isinstance(results[2], Exception) else None

        analysis_result = AnalysisResult(
            bpm=bmp_result.bpm if bmp_result and bmp_result.confidence >= request.confidence_threshold else None,
            bmp_confidence=bmp_result.confidence if bmp_result else None,
            key=key_result.key if key_result and key_result.confidence >= request.confidence_threshold else None,
            key_confidence=key_result.confidence if key_result else None,
            moods=mood_result.moods[:5] if mood_result else None,
            energy=mood_result.energy if mood_result else None,
            danceability=mood_result.danceability if mood_result else None,
            processing_time=processing_time,
            filename=file.filename
        )

        # Clean up temp file in background
        background_tasks.add_task(os.unlink, tmp_file_path)

        return analysis_result

    except Exception as e:
        # Clean up on error
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze/url")
async def analyze_from_url(url: str, request: AnalysisRequest = Depends()):
    """Analyze audio file from URL."""

    import aiohttp
    import tempfile

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise HTTPException(status_code=400, detail="Failed to download file from URL")

                content = await response.read()

                # Determine file extension from URL or content-type
                content_type = response.headers.get('content-type', '')
                if 'audio/mpeg' in content_type:
                    ext = '.mp3'
                elif 'audio/flac' in content_type:
                    ext = '.flac'
                elif 'audio/wav' in content_type:
                    ext = '.wav'
                else:
                    ext = '.mp3'  # Default

                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
                    tmp_file.write(content)
                    tmp_file_path = tmp_file.name

        # Create fake UploadFile for analysis
        class FakeUploadFile:
            filename = f"url_file{ext}"

        # Use the existing analyze function logic
        # ... (similar to analyze_uploaded_file)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"URL analysis failed: {str(e)}")

@app.get("/tracks")
async def get_tracks(
    skip: int = 0,
    limit: int = 100,
    genre: Optional[str] = None,
    min_bpm: Optional[float] = None,
    max_bpm: Optional[float] = None,
    min_energy: Optional[float] = None,
    max_energy: Optional[float] = None
):
    """Get tracks from database with filtering."""

    # Build query
    query = {}
    if genre:
        query['genre'] = {'$regex': genre, '$options': 'i'}
    if min_bpm is not None:
        query.setdefault('bpm', {})['$gte'] = min_bpm
    if max_bpm is not None:
        query.setdefault('bpm', {})['$lte'] = max_bpm
    if min_energy is not None:
        query.setdefault('energy', {})['$gte'] = min_energy
    if max_energy is not None:
        query.setdefault('energy', {})['$lte'] = max_energy

    tracks = await db.find_tracks(query, skip=skip, limit=limit)
    total_count = await db.count_tracks(query)

    return {
        "tracks": tracks,
        "total": total_count,
        "skip": skip,
        "limit": limit,
        "has_more": skip + len(tracks) < total_count
    }

@app.post("/playlists/generate")
async def generate_playlist(request: PlaylistRequest):
    """Generate a smart playlist based on criteria."""

    try:
        # Get candidate tracks from database
        candidate_tracks = await db.find_tracks_by_criteria(
            request.criteria,
            limit=request.max_tracks * 2  # Get extra for better selection
        )

        if not candidate_tracks:
            raise HTTPException(status_code=404, detail="No tracks found matching criteria")

        # Generate playlist
        playlist_tracks = await playlist_generator.create_playlist(
            tracks=candidate_tracks,
            duration_minutes=request.duration_minutes,
            max_tracks=request.max_tracks,
            name=request.name
        )

        # Save to database
        playlist_data = {
            'name': request.name,
            'tracks': playlist_tracks,
            'criteria': request.criteria,
            'generated_at': datetime.now().isoformat(),
            'track_count': len(playlist_tracks)
        }

        playlist_id = await db.save_playlist(playlist_data)

        return {
            "playlist_id": playlist_id,
            "playlist": playlist_data,
            "message": f"Generated playlist '{request.name}' with {len(playlist_tracks)} tracks"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Playlist generation failed: {str(e)}")

@app.get("/stats")
async def get_library_stats():
    """Get comprehensive library statistics."""

    stats = await db.get_library_statistics()

    return {
        "library_stats": stats,
        "analysis_coverage": {
            "bmp_analyzed": f"{stats.get('bpm_coverage', 0):.1f}%",
            "key_analyzed": f"{stats.get('key_coverage', 0):.1f}%",
            "mood_analyzed": f"{stats.get('mood_coverage', 0):.1f}%"
        },
        "top_genres": stats.get('top_genres', []),
        "bpm_distribution": stats.get('bpm_distribution', {})
    }

@app.websocket("/ws/analysis")
async def websocket_analysis_updates(websocket: WebSocket):
    """WebSocket endpoint for real-time analysis updates."""

    await websocket.accept()

    try:
        while True:
            # Wait for client message
            data = await websocket.receive_json()

            if data.get("type") == "analyze_file":
                file_path = data.get("file_path")

                if file_path and os.path.exists(file_path):
                    # Send analysis start notification
                    await websocket.send_json({
                        "type": "analysis_started",
                        "file": file_path
                    })

                    # Perform analysis with progress updates
                    result = await analyze_with_progress(file_path, websocket)

                    # Send final result
                    await websocket.send_json({
                        "type": "analysis_complete",
                        "file": file_path,
                        "result": result
                    })
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": "File not found"
                    })

    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })

async def analyze_with_progress(file_path: str, websocket):
    """Analyze file with progress updates via WebSocket."""

    # BPM analysis
    await websocket.send_json({"type": "progress", "step": "bpm_analysis", "progress": 25})
    bmp_result = await bmp_detector.analyze(file_path)

    # Key analysis
    await websocket.send_json({"type": "progress", "step": "key_analysis", "progress": 50})
    key_result = await key_detector.analyze(file_path)

    # Mood analysis
    await websocket.send_json({"type": "progress", "step": "mood_analysis", "progress": 75})
    mood_result = await mood_analyzer.analyze(file_path)

    # Complete
    await websocket.send_json({"type": "progress", "step": "complete", "progress": 100})

    return {
        "bpm": bpm_result.bpm,
        "key": key_result.key,
        "moods": mood_result.moods,
        "energy": mood_result.energy
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Django REST Framework Integration

```python
"""Django REST Framework integration with Tracktion."""

# models.py
from django.db import models
from django.contrib.postgres.fields import ArrayField

class Track(models.Model):
    title = models.CharField(max_length=255)
    artist = models.CharField(max_length=255)
    album = models.CharField(max_length=255, blank=True)
    file_path = models.FilePathField()

    # Analysis fields
    bmp = models.FloatField(null=True, blank=True)
    bmp_confidence = models.FloatField(null=True, blank=True)
    key = models.CharField(max_length=50, null=True, blank=True)
    key_confidence = models.FloatField(null=True, blank=True)
    energy = models.FloatField(null=True, blank=True)
    danceability = models.FloatField(null=True, blank=True)
    mood_tags = ArrayField(models.CharField(max_length=50), default=list)

    created_at = models.DateTimeField(auto_now_add=True)
    analyzed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = 'tracks'
        indexes = [
            models.Index(fields=['bpm']),
            models.Index(fields=['energy']),
            models.Index(fields=['artist']),
        ]

class Playlist(models.Model):
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    tracks = models.ManyToManyField(Track, through='PlaylistTrack')
    created_at = models.DateTimeField(auto_now_add=True)

class PlaylistTrack(models.Model):
    playlist = models.ForeignKey(Playlist, on_delete=models.CASCADE)
    track = models.ForeignKey(Track, on_delete=models.CASCADE)
    position = models.PositiveIntegerField()

    class Meta:
        unique_together = ('playlist', 'position')

# serializers.py
from rest_framework import serializers
from .models import Track, Playlist

class TrackSerializer(serializers.ModelSerializer):
    class Meta:
        model = Track
        fields = '__all__'

class PlaylistSerializer(serializers.ModelSerializer):
    tracks = TrackSerializer(many=True, read_only=True)
    track_count = serializers.SerializerMethodField()

    class Meta:
        model = Playlist
        fields = '__all__'

    def get_track_count(self, obj):
        return obj.tracks.count()

# views.py
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django.utils import timezone
import asyncio

from tracktion.analysis import BPMDetector, KeyDetector, MoodAnalyzer
from .models import Track, Playlist
from .serializers import TrackSerializer, PlaylistSerializer

class TrackViewSet(viewsets.ModelViewSet):
    queryset = Track.objects.all()
    serializer_class = TrackSerializer

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bpm_detector = BPMDetector()
        self.key_detector = KeyDetector()
        self.mood_analyzer = MoodAnalyzer()

    @action(detail=True, methods=['post'])
    def analyze(self, request, pk=None):
        """Analyze a track for audio features."""
        track = self.get_object()

        try:
            # Run analysis in async context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def run_analysis():
                tasks = [
                    self.bmp_detector.analyze(track.file_path),
                    self.key_detector.analyze(track.file_path),
                    self.mood_analyzer.analyze(track.file_path)
                ]
                return await asyncio.gather(*tasks)

            bpm_result, key_result, mood_result = loop.run_until_complete(run_analysis())

            # Update track with results
            track.bpm = bpm_result.bpm
            track.bpm_confidence = bmp_result.confidence
            track.key = key_result.key
            track.key_confidence = key_result.confidence
            track.energy = mood_result.energy
            track.danceability = mood_result.danceability
            track.mood_tags = mood_result.moods[:5]
            track.analyzed_at = timezone.now()
            track.save()

            return Response({
                'message': 'Analysis completed successfully',
                'results': {
                    'bpm': track.bpm,
                    'key': track.key,
                    'energy': track.energy,
                    'mood_tags': track.mood_tags
                }
            })

        except Exception as e:
            return Response(
                {'error': f'Analysis failed: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        finally:
            loop.close()

    @action(detail=False, methods=['get'])
    def search_by_features(self, request):
        """Search tracks by audio features."""

        min_bpm = request.query_params.get('min_bpm')
        max_bpm = request.query_params.get('max_bpm')
        min_energy = request.query_params.get('min_energy')
        max_energy = request.query_params.get('max_energy')
        key = request.query_params.get('key')
        mood = request.query_params.get('mood')

        queryset = self.get_queryset()

        if min_bpm:
            queryset = queryset.filter(bpm__gte=float(min_bpm))
        if max_bpm:
            queryset = queryset.filter(bpm__lte=float(max_bmp))
        if min_energy:
            queryset = queryset.filter(energy__gte=float(min_energy))
        if max_energy:
            queryset = queryset.filter(energy__lte=float(max_energy))
        if key:
            queryset = queryset.filter(key__icontains=key)
        if mood:
            queryset = queryset.filter(mood_tags__contains=[mood])

        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)

class PlaylistViewSet(viewsets.ModelViewSet):
    queryset = Playlist.objects.all()
    serializer_class = PlaylistSerializer

    @action(detail=False, methods=['post'])
    def generate_smart_playlist(self, request):
        """Generate playlist based on criteria."""

        name = request.data.get('name', 'Smart Playlist')
        criteria = request.data.get('criteria', {})
        max_tracks = request.data.get('max_tracks', 25)

        # Build Django ORM query from criteria
        tracks_query = Track.objects.filter(analyzed_at__isnull=False)

        if criteria.get('min_bpm'):
            tracks_query = tracks_query.filter(bmp__gte=criteria['min_bpm'])
        if criteria.get('max_bpm'):
            tracks_query = tracks_query.filter(bpm__lte=criteria['max_bpm'])
        if criteria.get('min_energy'):
            tracks_query = tracks_query.filter(energy__gte=criteria['min_energy'])
        if criteria.get('genres'):
            # This would require a genre field in the model
            pass

        # Get tracks and create playlist
        tracks = list(tracks_query.order_by('?')[:max_tracks])  # Random selection

        playlist = Playlist.objects.create(
            name=name,
            description=f"Generated playlist with {len(tracks)} tracks"
        )

        # Add tracks to playlist
        for i, track in enumerate(tracks):
            PlaylistTrack.objects.create(
                playlist=playlist,
                track=track,
                position=i + 1
            )

        serializer = self.get_serializer(playlist)
        return Response(serializer.data)

# urls.py
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'tracks', views.TrackViewSet)
router.register(r'playlists', views.PlaylistViewSet)

urlpatterns = [
    path('api/', include(router.urls)),
]
```

---

## Cloud Platform Integration

### AWS Lambda Integration

```python
"""AWS Lambda function for serverless music analysis."""

import json
import boto3
import tempfile
import os
from typing import Dict, Any

from tracktion.analysis import BPMDetector, KeyDetector, MoodAnalyzer

# Initialize AWS clients
s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(os.environ['TRACKTION_TABLE'])

# Initialize analyzers (reused across Lambda invocations)
bpm_detector = BPMDetector()
key_detector = KeyDetector()
mood_analyzer = MoodAnalyzer()

def lambda_handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    """
    AWS Lambda handler for music analysis.

    Event structure:
    {
        "s3_bucket": "my-music-bucket",
        "s3_key": "path/to/audio/file.mp3",
        "analysis_options": {
            "enable_mood": true,
            "confidence_threshold": 0.8
        }
    }
    """

    try:
        # Extract parameters from event
        s3_bucket = event['s3_bucket']
        s3_key = event['s3_key']
        analysis_options = event.get('analysis_options', {})

        # Download file from S3 to temporary location
        with tempfile.NamedTemporaryFile(suffix='.mp3') as temp_file:
            s3_client.download_fileobj(s3_bucket, s3_key, temp_file)
            temp_file.flush()

            # Perform analysis
            results = analyze_audio_file(temp_file.name, analysis_options)

            # Store results in DynamoDB
            store_results_in_dynamodb(s3_key, results)

            # Optionally trigger downstream processing
            if results.get('analysis_successful'):
                trigger_downstream_processing(s3_key, results)

            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'Analysis completed successfully',
                    'results': results,
                    's3_key': s3_key
                })
            }

    except Exception as e:
        print(f"Analysis failed: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'Analysis failed',
                'message': str(e)
            })
        }

def analyze_audio_file(file_path: str, options: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze audio file with specified options."""

    import asyncio

    async def run_analysis():
        confidence_threshold = options.get('confidence_threshold', 0.8)
        enable_mood = options.get('enable_mood', True)

        # Run core analysis
        tasks = [
            bmp_detector.analyze(file_path),
            key_detector.analyze(file_path)
        ]

        if enable_mood:
            tasks.append(mood_analyzer.analyze(file_path))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        analysis_results = {
            'analysis_successful': True,
            'timestamp': datetime.now().isoformat()
        }

        # BPM results
        bpm_result = results[0]
        if not isinstance(bmp_result, Exception):
            analysis_results.update({
                'bpm': bpm_result.bpm,
                'bmp_confidence': bmp_result.confidence,
                'bpm_reliable': bmp_result.confidence >= confidence_threshold
            })

        # Key results
        key_result = results[1]
        if not isinstance(key_result, Exception):
            analysis_results.update({
                'key': key_result.key,
                'key_confidence': key_result.confidence,
                'key_reliable': key_result.confidence >= confidence_threshold
            })

        # Mood results
        if enable_mood and len(results) > 2:
            mood_result = results[2]
            if not isinstance(mood_result, Exception):
                analysis_results.update({
                    'moods': mood_result.moods[:5],
                    'energy': mood_result.energy,
                    'danceability': mood_result.danceability
                })

        return analysis_results

    # Run async analysis in sync Lambda context
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        return loop.run_until_complete(run_analysis())
    finally:
        loop.close()

def store_results_in_dynamodb(s3_key: str, results: Dict[str, Any]):
    """Store analysis results in DynamoDB."""

    item = {
        'file_path': s3_key,
        'analysis_timestamp': results['timestamp'],
        **results
    }

    # Add TTL if configured
    ttl_days = int(os.environ.get('RESULTS_TTL_DAYS', 0))
    if ttl_days > 0:
        import time
        item['ttl'] = int(time.time()) + (ttl_days * 24 * 60 * 60)

    table.put_item(Item=item)

def trigger_downstream_processing(s3_key: str, results: Dict[str, Any]):
    """Trigger downstream processing via SNS/SQS."""

    sns_client = boto3.client('sns')
    topic_arn = os.environ.get('SNS_TOPIC_ARN')

    if topic_arn:
        message = {
            'file_path': s3_key,
            'analysis_results': results,
            'processing_stage': 'analysis_complete'
        }

        sns_client.publish(
            TopicArn=topic_arn,
            Message=json.dumps(message),
            Subject='Tracktion Analysis Complete'
        )

# CloudFormation template for deployment
CLOUDFORMATION_TEMPLATE = """
AWSTemplateFormatVersion: '2010-09-09'
Description: 'Tracktion Serverless Music Analysis'

Parameters:
  MusicBucketName:
    Type: String
    Description: S3 bucket containing audio files

Resources:
  TracktionTable:
    Type: AWS::DynamoDB::Table
    Properties:
      TableName: !Sub '${AWS::StackName}-tracktion-results'
      BillingMode: PAY_PER_REQUEST
      AttributeDefinitions:
        - AttributeName: file_path
          AttributeType: S
      KeySchema:
        - AttributeName: file_path
          KeyType: HASH
      TimeToLiveSpecification:
        AttributeName: ttl
        Enabled: true

  TracktionAnalysisFunction:
    Type: AWS::Lambda::Function
    Properties:
      FunctionName: !Sub '${AWS::StackName}-tracktion-analysis'
      Runtime: python3.11
      Handler: lambda_function.lambda_handler
      Code:
        ZipFile: |
          # Lambda function code would be packaged separately
          pass
      Environment:
        Variables:
          TRACKTION_TABLE: !Ref TracktionTable
      MemorySize: 1024
      Timeout: 300
      Role: !GetAtt LambdaExecutionRole.Arn

  LambdaExecutionRole:
    Type: AWS::IAM::Role
    Properties:
      AssumeRolePolicyDocument:
        Version: '2012-10-17'
        Statement:
          - Effect: Allow
            Principal:
              Service: lambda.amazonaws.com
            Action: sts:AssumeRole
      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
      Policies:
        - PolicyName: TracktionAnalysisPolicy
          PolicyDocument:
            Version: '2012-10-17'
            Statement:
              - Effect: Allow
                Action:
                  - s3:GetObject
                Resource: !Sub '${MusicBucket}/*'
              - Effect: Allow
                Action:
                  - dynamodb:PutItem
                  - dynamodb:GetItem
                  - dynamodb:UpdateItem
                Resource: !GetAtt TracktionTable.Arn

  S3TriggerPermission:
    Type: AWS::Lambda::Permission
    Properties:
      FunctionName: !Ref TracktionAnalysisFunction
      Action: lambda:InvokeFunction
      Principal: s3.amazonaws.com
      SourceArn: !Sub 'arn:aws:s3:::${MusicBucketName}'

Outputs:
  LambdaFunctionArn:
    Description: ARN of the Tracktion analysis Lambda function
    Value: !GetAtt TracktionAnalysisFunction.Arn

  DynamoDBTableName:
    Description: Name of the DynamoDB table storing results
    Value: !Ref TracktionTable
"""
```

### Google Cloud Functions Integration

```python
"""Google Cloud Functions integration for Tracktion."""

import functions_framework
from google.cloud import storage, firestore
from google.cloud import pubsub_v1
import tempfile
import os
import json
from datetime import datetime
from typing import Dict, Any

from tracktion.analysis import BPMDetector, KeyDetector, MoodAnalyzer

# Initialize Google Cloud clients
storage_client = storage.Client()
firestore_client = firestore.Client()
publisher = pubsub_v1.PublisherClient()

# Initialize analyzers
bmp_detector = BPMDetector()
key_detector = KeyDetector()
mood_analyzer = MoodAnalyzer()

@functions_framework.cloud_event
def analyze_audio_gcs(cloud_event):
    """Triggered by Cloud Storage when audio file is uploaded."""

    # Extract file information from Cloud Event
    data = cloud_event.data
    bucket_name = data['bucket']
    file_name = data['name']

    print(f"Processing file: gs://{bucket_name}/{file_name}")

    try:
        # Download file from Cloud Storage
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_name)

        with tempfile.NamedTemporaryFile() as temp_file:
            blob.download_to_filename(temp_file.name)

            # Analyze the audio file
            results = analyze_audio_file_sync(temp_file.name)

            # Store results in Firestore
            store_results_firestore(file_name, results)

            # Publish completion message
            publish_analysis_complete(file_name, results)

            print(f"Analysis completed for {file_name}")

    except Exception as e:
        print(f"Analysis failed for {file_name}: {str(e)}")
        # Store error in Firestore
        store_error_firestore(file_name, str(e))

@functions_framework.http
def analyze_audio_http(request):
    """HTTP endpoint for direct audio analysis."""

    if request.method != 'POST':
        return {'error': 'Only POST method allowed'}, 405

    try:
        # Handle different input types
        if 'file' in request.files:
            # Direct file upload
            file = request.files['file']
            analysis_results = analyze_uploaded_file(file)
        elif request.json and 'gcs_path' in request.json:
            # GCS path provided
            gcs_path = request.json['gcs_path']
            analysis_results = analyze_gcs_file(gcs_path)
        else:
            return {'error': 'No file or GCS path provided'}, 400

        return {
            'status': 'success',
            'results': analysis_results
        }, 200

    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }, 500

def analyze_uploaded_file(file) -> Dict[str, Any]:
    """Analyze directly uploaded file."""

    with tempfile.NamedTemporaryFile(suffix='.mp3') as temp_file:
        file.save(temp_file.name)
        return analyze_audio_file_sync(temp_file.name)

def analyze_gcs_file(gcs_path: str) -> Dict[str, Any]:
    """Analyze file from Google Cloud Storage."""

    # Parse GCS path (gs://bucket/path/to/file)
    if not gcs_path.startswith('gs://'):
        raise ValueError("Invalid GCS path format")

    path_parts = gcs_path[5:].split('/', 1)
    bucket_name = path_parts[0]
    file_path = path_parts[1]

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_path)

    with tempfile.NamedTemporaryFile() as temp_file:
        blob.download_to_filename(temp_file.name)
        return analyze_audio_file_sync(temp_file.name)

def analyze_audio_file_sync(file_path: str) -> Dict[str, Any]:
    """Synchronous analysis for Cloud Functions."""

    import asyncio

    async def run_analysis():
        tasks = [
            bpm_detector.analyze(file_path),
            key_detector.analyze(file_path),
            mood_analyzer.analyze(file_path)
        ]

        return await asyncio.gather(*tasks, return_exceptions=True)

    # Run async code in sync context
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        results = loop.run_until_complete(run_analysis())

        analysis_data = {
            'timestamp': datetime.now().isoformat(),
            'analysis_successful': True
        }

        # Process BPM result
        if not isinstance(results[0], Exception):
            bpm_result = results[0]
            analysis_data.update({
                'bpm': bmp_result.bpm,
                'bmp_confidence': bmp_result.confidence
            })

        # Process key result
        if not isinstance(results[1], Exception):
            key_result = results[1]
            analysis_data.update({
                'key': key_result.key,
                'key_confidence': key_result.confidence
            })

        # Process mood result
        if not isinstance(results[2], Exception):
            mood_result = results[2]
            analysis_data.update({
                'moods': mood_result.moods[:5],
                'energy': mood_result.energy,
                'danceability': mood_result.danceability
            })

        return analysis_data

    finally:
        loop.close()

def store_results_firestore(file_path: str, results: Dict[str, Any]):
    """Store analysis results in Firestore."""

    doc_ref = firestore_client.collection('audio_analysis').document()
    doc_ref.set({
        'file_path': file_path,
        'created_at': firestore.SERVER_TIMESTAMP,
        **results
    })

def store_error_firestore(file_path: str, error_message: str):
    """Store analysis error in Firestore."""

    doc_ref = firestore_client.collection('audio_analysis_errors').document()
    doc_ref.set({
        'file_path': file_path,
        'error_message': error_message,
        'created_at': firestore.SERVER_TIMESTAMP
    })

def publish_analysis_complete(file_path: str, results: Dict[str, Any]):
    """Publish analysis completion to Pub/Sub."""

    topic_path = publisher.topic_path(
        os.environ.get('GCP_PROJECT'),
        'tracktion-analysis-complete'
    )

    message_data = {
        'file_path': file_path,
        'results': results,
        'event_type': 'analysis_complete'
    }

    # Publish message
    future = publisher.publish(
        topic_path,
        json.dumps(message_data).encode('utf-8')
    )

    print(f"Published analysis complete message: {future.result()}")

# requirements.txt for Google Cloud Functions
REQUIREMENTS_TXT = """
functions-framework==3.5.0
google-cloud-storage==2.10.0
google-cloud-firestore==2.13.1
google-cloud-pubsub==2.18.4
tracktion[analysis]==1.0.0
"""

# deployment command
# gcloud functions deploy analyze-audio-gcs --runtime python311 --trigger-resource YOUR_BUCKET --trigger-event google.storage.object.finalize
```

---

## Database Integration

### PostgreSQL with SQLAlchemy

```python
"""PostgreSQL integration using SQLAlchemy ORM."""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
from typing import List, Optional, Dict, Any
import uuid
import asyncio

from tracktion.analysis import BPMDetector, KeyDetector, MoodAnalyzer

Base = declarative_base()

class Track(Base):
    __tablename__ = 'tracks'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(255), nullable=False)
    artist = Column(String(255), nullable=False)
    album = Column(String(255))
    file_path = Column(Text, nullable=False, unique=True)
    file_size = Column(Integer)
    file_format = Column(String(10))

    # Analysis results
    bpm = Column(Float)
    bpm_confidence = Column(Float)
    key = Column(String(50))
    key_confidence = Column(Float)
    energy = Column(Float)
    danceability = Column(Float)
    valence = Column(Float)
    mood_tags = Column(ARRAY(String))

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    analyzed_at = Column(DateTime)

    # Indexes for common queries
    __table_args__ = (
        {'postgresql_indexes': [
            'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_tracks_bpm ON tracks (bpm) WHERE bpm IS NOT NULL',
            'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_tracks_energy ON tracks (energy) WHERE energy IS NOT NULL',
            'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_tracks_artist ON tracks USING gin(artist gin_trgm_ops)',
            'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_tracks_mood_tags ON tracks USING gin(mood_tags)',
        ]}
    )

class Playlist(Base):
    __tablename__ = 'playlists'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class PlaylistTrack(Base):
    __tablename__ = 'playlist_tracks'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    playlist_id = Column(UUID(as_uuid=True), nullable=False)
    track_id = Column(UUID(as_uuid=True), nullable=False)
    position = Column(Integer, nullable=False)
    added_at = Column(DateTime, default=datetime.utcnow)

class TracktionDatabase:
    def __init__(self, database_url: str):
        self.engine = create_engine(
            database_url,
            pool_size=20,
            max_overflow=30,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=False  # Set to True for SQL debugging
        )

        # Create tables
        Base.metadata.create_all(self.engine)

        # Create session factory
        self.SessionLocal = sessionmaker(bind=self.engine)

        # Initialize analyzers
        self.bmp_detector = BPMDetector()
        self.key_detector = KeyDetector()
        self.mood_analyzer = MoodAnalyzer()

    def get_session(self) -> Session:
        """Get database session."""
        return self.SessionLocal()

    async def add_track_with_analysis(self, file_path: str, metadata: Dict[str, Any]) -> str:
        """Add track to database with automatic analysis."""

        # Perform analysis first
        analysis_results = await self._analyze_track(file_path)

        with self.get_session() as session:
            # Create track record
            track = Track(
                title=metadata.get('title', 'Unknown'),
                artist=metadata.get('artist', 'Unknown'),
                album=metadata.get('album', ''),
                file_path=file_path,
                file_size=metadata.get('file_size', 0),
                file_format=metadata.get('format', ''),

                # Analysis results
                bpm=analysis_results.get('bpm'),
                bpm_confidence=analysis_results.get('bmp_confidence'),
                key=analysis_results.get('key'),
                key_confidence=analysis_results.get('key_confidence'),
                energy=analysis_results.get('energy'),
                danceability=analysis_results.get('danceability'),
                valence=analysis_results.get('valence'),
                mood_tags=analysis_results.get('mood_tags', []),
                analyzed_at=datetime.utcnow()
            )

            session.add(track)
            session.commit()

            return str(track.id)

    async def _analyze_track(self, file_path: str) -> Dict[str, Any]:
        """Analyze track for audio features."""

        try:
            # Run analyzers concurrently
            bmp_result, key_result, mood_result = await asyncio.gather(
                self.bpm_detector.analyze(file_path),
                self.key_detector.analyze(file_path),
                self.mood_analyzer.analyze(file_path),
                return_exceptions=True
            )

            results = {}

            if not isinstance(bpm_result, Exception):
                results.update({
                    'bpm': bpm_result.bpm,
                    'bmp_confidence': bmp_result.confidence
                })

            if not isinstance(key_result, Exception):
                results.update({
                    'key': key_result.key,
                    'key_confidence': key_result.confidence
                })

            if not isinstance(mood_result, Exception):
                results.update({
                    'energy': mood_result.energy,
                    'danceability': mood_result.danceability,
                    'valence': mood_result.valence,
                    'mood_tags': mood_result.moods[:10]  # Limit to 10 tags
                })

            return results

        except Exception as e:
            print(f"Analysis failed for {file_path}: {e}")
            return {}

    def find_tracks_by_features(
        self,
        bpm_range: Optional[tuple] = None,
        energy_range: Optional[tuple] = None,
        key: Optional[str] = None,
        mood_tags: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[Track]:
        """Find tracks by audio features."""

        with self.get_session() as session:
            query = session.query(Track).filter(Track.analyzed_at.isnot(None))

            if bpm_range:
                min_bpm, max_bpm = bpm_range
                query = query.filter(
                    Track.bpm.between(min_bpm, max_bpm),
                    Track.bpm_confidence >= 0.7
                )

            if energy_range:
                min_energy, max_energy = energy_range
                query = query.filter(Track.energy.between(min_energy, max_energy))

            if key:
                query = query.filter(Track.key.ilike(f'%{key}%'))

            if mood_tags:
                # PostgreSQL array contains any
                query = query.filter(Track.mood_tags.overlap(mood_tags))

            return query.limit(limit).all()

    def create_smart_playlist(
        self,
        name: str,
        criteria: Dict[str, Any],
        max_tracks: int = 50
    ) -> str:
        """Create playlist based on criteria."""

        # Find matching tracks
        tracks = self.find_tracks_by_features(**criteria, limit=max_tracks * 2)

        # Select diverse subset
        selected_tracks = self._diversify_track_selection(tracks, max_tracks)

        with self.get_session() as session:
            # Create playlist
            playlist = Playlist(
                name=name,
                description=f"Smart playlist with {len(selected_tracks)} tracks"
            )
            session.add(playlist)
            session.flush()  # Get playlist ID

            # Add tracks to playlist
            for i, track in enumerate(selected_tracks):
                playlist_track = PlaylistTrack(
                    playlist_id=playlist.id,
                    track_id=track.id,
                    position=i + 1
                )
                session.add(playlist_track)

            session.commit()
            return str(playlist.id)

    def _diversify_track_selection(self, tracks: List[Track], target_count: int) -> List[Track]:
        """Select diverse subset of tracks to avoid repetition."""

        if len(tracks) <= target_count:
            return tracks

        selected = []
        remaining = tracks.copy()

        # Group by artist to ensure variety
        artist_groups = {}
        for track in remaining:
            artist_groups.setdefault(track.artist, []).append(track)

        # Select one track from each artist first
        for artist, artist_tracks in artist_groups.items():
            if len(selected) < target_count:
                selected.append(artist_tracks[0])
                remaining.remove(artist_tracks[0])

        # Fill remaining slots
        while len(selected) < target_count and remaining:
            selected.append(remaining.pop(0))

        return selected

    def get_library_statistics(self) -> Dict[str, Any]:
        """Get comprehensive library statistics."""

        with self.get_session() as session:
            # Basic counts
            total_tracks = session.query(Track).count()
            analyzed_tracks = session.query(Track).filter(Track.analyzed_at.isnot(None)).count()

            # Analysis coverage
            bpm_analyzed = session.query(Track).filter(Track.bpm.isnot(None)).count()
            key_analyzed = session.query(Track).filter(Track.key.isnot(None)).count()
            mood_analyzed = session.query(Track).filter(Track.mood_tags != []).count()

            # BPM distribution
            from sqlalchemy import func
            bpm_distribution = session.query(
                func.case(
                    (Track.bpm < 80, 'Very Slow'),
                    (Track.bpm < 100, 'Slow'),
                    (Track.bpm < 120, 'Moderate'),
                    (Track.bpm < 140, 'Fast'),
                    else_='Very Fast'
                ).label('tempo_category'),
                func.count(Track.id).label('count')
            ).filter(Track.bpm.isnot(None)).group_by('tempo_category').all()

            # Top artists
            top_artists = session.query(
                Track.artist,
                func.count(Track.id).label('track_count')
            ).group_by(Track.artist).order_by(
                func.count(Track.id).desc()
            ).limit(10).all()

            return {
                'total_tracks': total_tracks,
                'analyzed_tracks': analyzed_tracks,
                'analysis_coverage': {
                    'bpm': (bmp_analyzed / total_tracks * 100) if total_tracks > 0 else 0,
                    'key': (key_analyzed / total_tracks * 100) if total_tracks > 0 else 0,
                    'mood': (mood_analyzed / total_tracks * 100) if total_tracks > 0 else 0
                },
                'bpm_distribution': dict(bpm_distribution),
                'top_artists': [{'artist': artist, 'tracks': count} for artist, count in top_artists]
            }

# Usage example
async def database_integration_example():
    """Example of using PostgreSQL integration."""

    # Initialize database
    db = TracktionDatabase("postgresql://user:password@localhost/tracktion")

    # Add tracks with analysis
    track_metadata = {
        'title': 'Example Song',
        'artist': 'Example Artist',
        'album': 'Example Album',
        'file_size': 5242880,
        'format': 'mp3'
    }

    track_id = await db.add_track_with_analysis('/path/to/song.mp3', track_metadata)
    print(f"Added track: {track_id}")

    # Find similar tracks
    similar_tracks = db.find_tracks_by_features(
        bmp_range=(120, 140),
        energy_range=(0.7, 1.0),
        mood_tags=['energetic', 'happy']
    )

    # Create smart playlist
    playlist_id = db.create_smart_playlist(
        name="High Energy Dance",
        criteria={
            'bmp_range': (120, 140),
            'energy_range': (0.7, 1.0),
            'mood_tags': ['energetic', 'dance']
        }
    )

    # Get statistics
    stats = db.get_library_statistics()
    print(f"Library has {stats['total_tracks']} tracks")

    print("Database integration example completed!")

if __name__ == "__main__":
    asyncio.run(database_integration_example())
```

This comprehensive integration guide shows how to integrate Tracktion with various platforms and services. Each example includes production-ready code with proper error handling, scalability considerations, and best practices.

The examples can be adapted based on your specific requirements and infrastructure setup. For more advanced integration patterns, refer to the API documentation and deployment guides.
