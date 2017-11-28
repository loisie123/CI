# This is the file for the Particle Swarm Optimization.

# Initialize swarm
def startSwarm:

    return swarm

# Calculate fitness values
def particleFitness:

    return fitness

# Pick the best particle
def globalBest:

    return gbest

# Calculate velocity
def calculateVelocity:

    return velocity

# Update particle position
def updateParticles:

    return swarm

# Main function
def PSO_algorithm:

    # Initialize the swarm (if first iteration)
    starswarm()

    # Calculate fitness for every particle
    particleFitness()

    # Pick the best particle
    globalBest()

    # Calculate velocity for every particle
    calculateVelocity()

    # Update particle position and use this as new swarm
    updateParticles()

    return matrix
