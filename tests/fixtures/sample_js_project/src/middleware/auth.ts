export class AuthMiddleware {
  authenticate(token: string): boolean {
    if (!token) {
      throw new Error("Missing token");
    }
    return true;
  }

  validateExpiry(expTime: number): boolean {
    return Date.now() < expTime * 1000;
  }
}

export function verifySecret(secret: string): boolean {
  return secret.length >= 32;
}
