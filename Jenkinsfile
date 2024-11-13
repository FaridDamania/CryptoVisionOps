pipeline {
    agent any  // Use any available agent for the pipeline

    environment {
        DOCKER_IMAGE = "cryptovisionops"  // Define the Docker image name
        DOCKER_PORT = "8000"  // Define the port for the application
    }

    stages {
        stage('Checkout Code') {
            steps {
                // Checkout the code from the SCM configured in Jenkins
                echo 'Checking out the code...'
                checkout scm
            }
        }

        stage('Install Dependencies') {
            steps {
                echo 'Installing Python dependencies...'
                // Ensure Python dependencies are installed
                sh '''
                if [ -f requirements.txt ]; then
                    pip install --upgrade pip
                    pip install -r requirements.txt
                else
                    echo "No requirements.txt found!"
                    exit 1
                fi
                '''
            }
        }

        stage('Run Tests') {
            steps {
                echo 'Running test cases...'
                // Run the tests and ensure failures stop the pipeline
                sh '''
                if [ -d tests ]; then
                    pytest --maxfail=5 --disable-warnings
                else
                    echo "No tests directory found!"
                    exit 1
                fi
                '''
            }
        }

        stage('Build Docker Image') {
            steps {
                echo 'Building the Docker image...'
                // Build the Docker image
                sh '''
                docker build -t $DOCKER_IMAGE .
                docker image prune -f  # Clean up dangling images to save space
                '''
            }
        }

        stage('Deploy Application') {
            steps {
                echo 'Deploying the application...'
                // Deploy the application with Docker
                sh '''
                docker stop $DOCKER_IMAGE || true
                docker rm $DOCKER_IMAGE || true
                docker run -d --name $DOCKER_IMAGE -p $DOCKER_PORT:8000 $DOCKER_IMAGE
                '''
            }
        }
    }

    post {
        always {
            echo 'Pipeline execution completed.'
            // Clean up temporary files or artifacts if necessary
        }
        success {
            echo 'Pipeline executed successfully.'
        }
        failure {
            echo 'Pipeline failed. Check the logs for details.'
        }
    }
}
