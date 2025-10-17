import os 
import json
import time
import boto3
import sagemaker
from sagemaker.huggingface.model import HuggingFaceModel


model_artifact_s3_uri = os.environ.get("MODEL_ARTIFACT_S3_URI") 
sagemaker_role_arn = os.environ.get("SAGEMAKER_ROLE_ARN")    

endpoint_name = "gemma-3-12b-zero-scale-endpoint"
inference_component_name = "gemma-3-12b-component"
instance_type = "ml.g5.2xlarge" 
# --- END OF CONFIGURATION ---

# --- Script Validation ---
if not model_artifact_s3_uri or not sagemaker_role_arn:
    raise ValueError("Error: Environment variables MODEL_ARTIFACT_S3_URI and SAGEMAKER_ROLE_ARN must be set.")

region = boto3.Session().region_name

print(f"Using SageMaker Role: {sagemaker_role_arn}")
print(f"Using Model Artifact: {model_artifact_s3_uri}")
print(f"Using Region: {region}")

# Initialize boto3 clients
sm_client = boto3.client("sagemaker", region_name=region)
asg_client = boto3.client("application-autoscaling", region_name=region)
cw_client = boto3.client("cloudwatch", region_name=region)

# =====================================================================================
# DISCOVER OLD RESOURCES FOR CLEANUP
# =====================================================================================

print("\n[0/7] Discovering existing resources for cleanup...")
old_model_name = None
old_endpoint_config_name = None

try:
    endpoint_desc = sm_client.describe_endpoint(EndpointName=endpoint_name)
    old_endpoint_config_name = endpoint_desc['EndpointConfigName']
    print(f"   Found existing endpoint config: {old_endpoint_config_name}")

    config_desc = sm_client.describe_endpoint_config(EndpointConfigName=old_endpoint_config_name)
    # Note: Inference Components don't list models here, so we look for the model attached to the IC
    ic_desc = sm_client.describe_inference_component(InferenceComponentName=inference_component_name)
    old_model_name = ic_desc['Specification']['ModelName']
    print(f"   Found existing model to be replaced: {old_model_name}")

except sm_client.exceptions.ClientError as e:
    error_code = e.response['Error']['Code']
    if 'ValidationException' in error_code or 'ResourceNotFound' in error_code:
        print("   No existing endpoint found. Will proceed with a new creation.")
    else:
        raise # Re-raise other unexpected errors

# =====================================================================================
# CLEAN UP OLD, DETACHED RESOURCES
# =====================================================================================
print(f"\n[1/7] Cleaning up old resources...")

if old_model_name:
    try:
        sm_client.delete_model(ModelName=old_model_name)
        print(f"✅ Successfully deleted old model: {old_model_name}")
    except sm_client.exceptions.ClientError as e:
        print(f"⚠️  Could not delete old model '{old_model_name}'. It might already be gone. Error: {e}")

if old_endpoint_config_name:
    try:
        # Note: SageMaker might still be cleaning up the old EC, so this can sometimes fail.
        # In a production system, you might build a separate cleanup lambda. For this project, it's fine.
        sm_client.delete_endpoint_config(EndpointConfigName=old_endpoint_config_name)
        print(f"✅ Successfully deleted old endpoint config: {old_endpoint_config_name}")
    except sm_client.exceptions.ClientError as e:
        print(f"⚠️  Could not delete old endpoint config '{old_endpoint_config_name}'. It might already be gone. Error: {e}")


# =====================================================================================
# MODEL CREATION
# =====================================================================================

model_name = f"gemma-3-12b-model-{int(time.time())}"
print(f"\n[2/7] Creating SageMaker Model: {model_name}")

# Get the HuggingFace DLC image URI
huggingface_model = HuggingFaceModel(
    model_data=model_artifact_s3_uri,
    role=sagemaker_role_arn,
    transformers_version="4.37",
    pytorch_version="2.1",
    py_version="py310",
)

# Get the image URI for the container
image_uri = huggingface_model.serving_image_uri(
    region_name=region,
    instance_type=instance_type
)

print(f"Using image: {image_uri}")

# Create the model using boto3
create_model_response = sm_client.create_model(
    ModelName=model_name,
    PrimaryContainer={
        'Image': image_uri,
        'ModelDataUrl': model_artifact_s3_uri,
        'Environment': {
            'SAGEMAKER_CONTAINER_LOG_LEVEL': '20',
            'SAGEMAKER_REGION': region,
        }
    },
    ExecutionRoleArn=sagemaker_role_arn
)

print(f"✅ Model created: {model_name}")

# =====================================================================================
# CREATE ENDPOINT CONFIGURATION WITH MANAGED INSTANCE SCALING
# CRITICAL: Do NOT specify ModelName when using Inference Components!
# =====================================================================================

endpoint_config_name = f"gemma-3-12b-config-{int(time.time())}"
print(f"\n[3/7] Creating Endpoint Configuration: {endpoint_config_name}")
print(" Note: NOT specifying ModelName - this will be attached via Inference Component")

endpoint_config_response = sm_client.create_endpoint_config(
    EndpointConfigName=endpoint_config_name,
    ExecutionRoleArn=sagemaker_role_arn,
    ProductionVariants=[
        {
            'VariantName': 'AllTraffic',
            # DO NOT specify ModelName here when using inference components!
            'InstanceType': instance_type,
            'InitialInstanceCount': 1,  # Start with 1, will scale to 0
            'ManagedInstanceScaling': {
                'Status': 'ENABLED',
                'MinInstanceCount': 0,  # CRITICAL: This enables scale-to-zero
                'MaxInstanceCount': 2   # Maximum instances when scaling out
            },
            'RoutingConfig': {
                'RoutingStrategy': 'LEAST_OUTSTANDING_REQUESTS'
            }
        }
    ]
)

print(f"✅ Endpoint configuration created with MinInstanceCount=0")

# =====================================================================================
# CREATE ENDPOINT
# =====================================================================================

print(f"\n[4/7] Creating Endpoint: {endpoint_name}")

try:
    sm_client.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_config_name
    )
    print(f"✅ Endpoint creation initiated. Waiting for 'InService' status...")
    print(f"   (This takes 5-10 minutes...)")
    
    # Wait for endpoint to be in service
    waiter = sm_client.get_waiter('endpoint_in_service')
    waiter.wait(EndpointName=endpoint_name)
    print(f"✅ Endpoint is now InService!")
    
except sm_client.exceptions.ClientError as e:
    error_code = e.response['Error']['Code']
    if error_code == 'ValidationException' and 'already exists' in str(e):
        print(f"⚠️  Endpoint {endpoint_name} already exists. Using existing endpoint.")
        # Wait for it to be in service
        waiter = sm_client.get_waiter('endpoint_in_service')
        waiter.wait(EndpointName=endpoint_name)
    else:
        raise

# =====================================================================================
# CREATE INFERENCE COMPONENT
# This attaches the model to the endpoint with resource specifications
# =====================================================================================

print(f"\n[5/7] Creating Inference Component: {inference_component_name}")

try:
    ic_response = sm_client.create_inference_component(
        InferenceComponentName=inference_component_name,
        EndpointName=endpoint_name,
        VariantName='AllTraffic',
        Specification={
            'ModelName': model_name,  # Now we attach the model via the component
            'ComputeResourceRequirements': {
                'MinMemoryRequiredInMb': 12288,  # 12GB for Gemma 12B (adjust as needed)
                'NumberOfAcceleratorDevicesRequired': 1  # 1 GPU
            }
        },
        RuntimeConfig={
            'CopyCount': 1  # Number of model copies (will auto-scale to 0)
        }
    )
    print(f"✅ Inference component creation initiated")
    
    # Wait for inference component to be in service
    print("   Waiting for inference component to be InService...")
    for i in range(60):  # Wait up to 10 minutes
        try:
            ic_status = sm_client.describe_inference_component(
                InferenceComponentName=inference_component_name
            )
            status = ic_status['InferenceComponentStatus']
            
            if status == 'InService':
                print(f"✅ Inference component is InService!")
                break
            elif status in ['Failed', 'Deleting']:
                failure_reason = ic_status.get('FailureReason', 'Unknown')
                raise Exception(f"Inference component failed: {status} - {failure_reason}")
            
            print(f"   Status: {status}... (waiting)", end='\r')
            time.sleep(10)
        except KeyError:
            time.sleep(10)
            continue
        
except sm_client.exceptions.ClientError as e:
    error_code = e.response['Error']['Code']
    if error_code == 'ValidationException' and 'already exists' in str(e):
        print(f"⚠️  Inference component already exists. Continuing...")
    else:
        print(f"❌ Error creating inference component: {e}")
        print("\nCommon issues:")
        print("  • MinMemoryRequiredInMb too low for the model size")
        print("  • Instance type doesn't have enough memory/GPU")
        print("  • Model loading failed (check CloudWatch logs)")
        raise

# =====================================================================================
# CONFIGURE AUTO-SCALING (SCALE-IN TO ZERO)
# =====================================================================================

print(f"\n[6/7] Configuring auto-scaling to scale IN to zero...")

resource_id = f"inference-component/{inference_component_name}"

try:
    # Register as scalable target with MinCapacity=0
    asg_client.register_scalable_target(
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:inference-component:DesiredCopyCount",
        MinCapacity=0,  # CRITICAL: Enables scale to zero
        MaxCapacity=3   # Maximum copies when scaling out
    )
    print(f"✅ Registered scalable target with MinCapacity=0")
except asg_client.exceptions.ValidationException as e:
    if 'already exists' in str(e).lower():
        print(f"⚠️  Scalable target already registered")
    else:
        raise

# Apply target tracking policy for scale-in
# This will scale down to 0 after 15 minutes of no traffic
asg_client.put_scaling_policy(
    PolicyName=f"target-tracking-{inference_component_name}",
    ServiceNamespace="sagemaker",
    ResourceId=resource_id,
    ScalableDimension="sagemaker:inference-component:DesiredCopyCount",
    PolicyType="TargetTrackingScaling",
    TargetTrackingScalingPolicyConfiguration={
        "PredefinedMetricSpecification": {
            "PredefinedMetricType": "SageMakerInferenceComponentInvocationsPerCopy"
        },
        "TargetValue": 5.0,  # Scale when concurrent requests > 5 per copy
        "ScaleInCooldown": 900,   # Wait 15 minutes before scaling in (AWS recommended)
        "ScaleOutCooldown": 60    # Wait 1 minute before scaling out again
    }
)

print(f"✅ Target tracking policy applied for scale-in (15 min cooldown)")

# =====================================================================================
# CONFIGURE SCALE-OUT FROM ZERO (STEP SCALING + CLOUDWATCH ALARM)
# =====================================================================================

print(f"\n[7/7] Configuring auto-scaling to scale OUT from zero...")

# Create step scaling policy
step_policy_response = asg_client.put_scaling_policy(
    PolicyName=f"step-scaling-{inference_component_name}",
    ServiceNamespace="sagemaker",
    ResourceId=resource_id,
    ScalableDimension="sagemaker:inference-component:DesiredCopyCount",
    PolicyType="StepScaling",
    StepScalingPolicyConfiguration={
        "AdjustmentType": "ChangeInCapacity",
        "MetricAggregationType": "Maximum",
        "Cooldown": 60,
        "StepAdjustments": [
            {
                "MetricIntervalLowerBound": 0,
                "ScalingAdjustment": 1  # Add 1 copy when triggered
            }
        ]
    }
)

step_policy_arn = step_policy_response['PolicyARN']
print(f"✅ Step scaling policy created")

# Create CloudWatch alarm to trigger scale-out
alarm_name = f"scale-out-alarm-{inference_component_name}"

cw_client.put_metric_alarm(
    AlarmName=alarm_name,
    AlarmDescription="Alarm when endpoint invoked with 0 instances",
    AlarmActions=[step_policy_arn],
    MetricName="NoCapacityInvocationFailures",
    Namespace="AWS/SageMaker",
    Statistic="Sum",
    Dimensions=[
        {
            'Name': 'InferenceComponentName',
            'Value': inference_component_name
        }
    ],
    Period=60,
    EvaluationPeriods=1,
    DatapointsToAlarm=1,
    Threshold=1,
    ComparisonOperator="GreaterThanOrEqualToThreshold"
)

print(f"✅ CloudWatch alarm created: {alarm_name}")


print(" SUCCESS! Scale-to-zero deployment is complete!")
print(f"\n📋 Endpoint Details:")
print(f"   Endpoint Name: {endpoint_name}")
print(f"   Inference Component: {inference_component_name}")
print(f"   Instance Type: {instance_type}")
print(f"   Model Name: {model_name}")