import pygame
import sys

# Initialize Pygame
pygame.init()

# Screen setup
screen_width, screen_height = 800, 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Keyboard Joystick Simulation")

# Colors
white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)

# Circle setup (joystick representation)
circle_radius = 15
circle_position = [screen_width // 2, screen_height // 2]
speed = 5

# Clock for FPS
clock = pygame.time.Clock()

# Main game loop
running = True
while running:
    screen.fill(white)

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Key handling
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        circle_position[1] -= speed
        print("holi")
    if keys[pygame.K_DOWN]:
        circle_position[1] += speed
    if keys[pygame.K_LEFT]:
        circle_position[0] -= speed
    if keys[pygame.K_RIGHT]:
        circle_position[0] += speed

    # Ensure the circle stays within the screen
    circle_position[0] = max(circle_radius, min(screen_width - circle_radius, circle_position[0]))
    circle_position[1] = max(circle_radius, min(screen_height - circle_radius, circle_position[1]))

    # Draw the circle
    pygame.draw.circle(screen, red, circle_position, circle_radius)

    # Update display
    pygame.display.flip()

    # Cap the frame rate
    clock.tick(60)

pygame.quit()
sys.exit()