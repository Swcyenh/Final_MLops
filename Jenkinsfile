pipeline {
    agent any

    environment {
        DOCKER_IMAGE = 'maddox1311/image-detection-app' // Tên đầy đủ trên DockerHub
        DOCKER_TAG = 'latest'
        DOCKER_CONTAINER_NAME = 'image-detection-container'
    }

    stages {
        stage('Clone Repository') {
            steps {
                git 'https://github.com/Swcyenh/Final_MLops.git'
            }
        }

        stage('Build Docker Image') {
            steps {
                script {
                    docker.build("${DOCKER_IMAGE}:${DOCKER_TAG}")
                }
            }
        }

        stage('Run Tests') {
            steps {
                script {
                    docker.image("${DOCKER_IMAGE}:${DOCKER_TAG}").inside {
                        // Chạy unit tests
                        sh 'python -m unittest discover -s tests -p "*.py"'
                    }
                }
            }
        }

        stage('Push Docker Image') {
            steps {
                script {
                    docker.withRegistry('', 'dockerhub-credentials') {
                        docker.image("${DOCKER_IMAGE}:${DOCKER_TAG}").push()
                    }
                }
            }
        }

        stage('Deploy') {
            steps {
                script {
                    // Xóa container cũ nếu có
                    sh "docker stop ${DOCKER_CONTAINER_NAME} || true"
                    sh "docker rm ${DOCKER_CONTAINER_NAME} || true"

                    // Chạy container mới
                    sh "docker run -d --name ${DOCKER_CONTAINER_NAME} -p 8080:8080 ${DOCKER_IMAGE}:${DOCKER_TAG}"
                }
            }
        }
    }

    post {
        always {
            echo 'Cleaning up dangling images...'
            sh 'docker image prune -f || true'
        }
        success {
            echo 'Deployment Successful!'
        }
        failure {
            echo 'Deployment Failed!'
        }
    }
}
