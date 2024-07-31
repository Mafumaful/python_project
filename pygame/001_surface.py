import pygame

# Initialize Pygame
pygame.init()

# Create a window with a specific size
window_size = (800, 600)
screen = pygame.display.set_mode(window_size)

# Create a Surface with the same size as the window
surface = pygame.Surface(window_size)

# Fill the Surface with a color (e.g., white)
surface.fill((255, 255, 255))

# Draw a red rectangle on the Surface
pygame.draw.rect(surface, (255, 0, 0), (100, 100, 200, 150))

# Blit the Surface onto the screen (display it)
screen.blit(surface, (0, 0))

# Update the display to show the changes
pygame.display.flip()

# Wait for a few seconds before quitting
pygame.time.wait(3000)
pygame.quit()
