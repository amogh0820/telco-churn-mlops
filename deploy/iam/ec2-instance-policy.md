# EC2 instance role

Attach these two AWS-managed policies to the instance profile on the EC2 box.
No inline policy is needed.

| Policy | Why |
|---|---|
| `AmazonEC2ContainerRegistryReadOnly` | lets the instance `docker pull` from ECR |
| `AmazonSSMManagedInstanceCore` | lets Systems Manager reach the instance, for both deploys and Session Manager shells |

`AmazonEC2RoleforSSM` is the deprecated predecessor of the second one. Do not
use it.
