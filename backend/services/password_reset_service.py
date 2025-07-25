# services/password_reset_service.py
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from sqlalchemy.orm import Session
from database.models import PasswordResetToken, User
from passlib.context import CryptContext
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class PasswordResetService:
    def __init__(self):
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        
        # Email configuration from environment variables
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_username = os.getenv("SMTP_USERNAME")
        self.smtp_password = os.getenv("SMTP_PASSWORD")
        self.from_email = os.getenv("FROM_EMAIL", self.smtp_username)
        self.frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")
        
    def initiate_password_reset(self, email: str, db: Session) -> bool:
        """
        Initiate password reset process by creating token and sending email
        
        Args:
            email: User's email address
            db: Database session
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if user exists
            user = db.query(User).filter(User.email == email.lower()).first()
            if not user:
                # For security, don't reveal if email exists or not
                logger.warning(f"Password reset attempted for non-existent email: {email}")
                return True  # Return True to not reveal email existence
            
            # Invalidate any existing tokens for this user
            existing_tokens = db.query(PasswordResetToken).filter(
                PasswordResetToken.user_id == user.id,
                PasswordResetToken.used == False
            ).all()
            
            for token in existing_tokens:
                token.mark_as_used()
            
            # Create new reset token
            reset_token = PasswordResetToken.create_token(user.id, email.lower())
            db.add(reset_token)
            db.commit()
            
            # Send reset email
            success = self._send_reset_email(email, reset_token.token)
            
            if success:
                logger.info(f"Password reset email sent successfully to {email}")
                return True
            else:
                # If email fails, mark token as used to prevent abuse
                reset_token.mark_as_used()
                db.commit()
                return False
                
        except Exception as e:
            logger.error(f"Error initiating password reset for {email}: {str(e)}")
            db.rollback()
            return False
    
    def reset_password(self, token: str, new_password: str, db: Session) -> dict:
        """
        Reset password using provided token
        
        Args:
            token: Password reset token
            new_password: New password
            db: Database session
            
        Returns:
            dict: Result with success status and message
        """
        try:
            # Find and validate token
            reset_token = db.query(PasswordResetToken).filter(
                PasswordResetToken.token == token
            ).first()
            
            if not reset_token:
                return {"success": False, "message": "Invalid reset token"}
            
            if not reset_token.is_valid():
                return {"success": False, "message": "Reset token has expired or been used"}
            
            # Find user
            user = db.query(User).filter(User.id == reset_token.user_id).first()
            if not user:
                return {"success": False, "message": "User not found"}
            
            # Validate new password
            if not self._validate_password(new_password):
                return {
                    "success": False, 
                    "message": "Password must be at least 8 characters long and contain uppercase, lowercase, number, and special character"
                }
            
            # Update user password
            hashed_password = self.pwd_context.hash(new_password)
            user.hashed_password = hashed_password
            user.updated_at = datetime.utcnow()
            
            # Mark token as used
            reset_token.mark_as_used()
            
            db.commit()
            
            logger.info(f"Password successfully reset for user: {user.email}")
            return {"success": True, "message": "Password reset successfully"}
            
        except Exception as e:
            logger.error(f"Error resetting password: {str(e)}")
            db.rollback()
            return {"success": False, "message": "An error occurred while resetting password"}
    
    def verify_reset_token(self, token: str, db: Session) -> dict:
        """
        Verify if a reset token is valid without using it
        
        Args:
            token: Password reset token
            db: Database session
            
        Returns:
            dict: Verification result
        """
        try:
            reset_token = db.query(PasswordResetToken).filter(
                PasswordResetToken.token == token
            ).first()
            
            if not reset_token:
                return {"valid": False, "message": "Invalid token"}
            
            if not reset_token.is_valid():
                return {"valid": False, "message": "Token has expired or been used"}
            
            return {
                "valid": True, 
                "email": reset_token.email,
                "expires_at": reset_token.expires_at.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error verifying reset token: {str(e)}")
            return {"valid": False, "message": "Error verifying token"}
    
    def _send_reset_email(self, email: str, token: str) -> bool:
        """
        Send password reset email
        
        Args:
            email: Recipient email
            token: Reset token
            
        Returns:
            bool: True if email sent successfully
        """
        try:
            if not all([self.smtp_username, self.smtp_password]):
                logger.error("SMTP credentials not configured")
                return False
            
            # Create reset URL
            reset_url = f"{self.frontend_url}/reset-password?token={token}"
            
            # Create email content
            subject = "WealthWise Password Reset Request"
            
            html_body = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Password Reset</title>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; margin: 0; padding: 0; }}
                    .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                    .header {{ background-color: #00A8FF; padding: 20px; text-align: center; border-radius: 8px 8px 0 0; }}
                    .header h1 {{ color: white; margin: 0; font-size: 24px; }}
                    .content {{ background-color: #ffffff; padding: 30px; border: 1px solid #e9ecef; }}
                    .footer {{ background-color: #f8f9fa; padding: 20px; text-align: center; font-size: 12px; color: #6c757d; border-radius: 0 0 8px 8px; }}
                    .button {{ display: inline-block; padding: 15px 30px; background-color: #00A8FF; color: white; text-decoration: none; border-radius: 5px; margin: 20px 0; font-weight: bold; }}
                    .button:hover {{ background-color: #0088CC; }}
                    .warning {{ background-color: #fff3cd; border: 1px solid: #ffeaa7; padding: 15px; border-radius: 4px; margin: 20px 0; }}
                    .url-box {{ background-color: #f8f9fa; padding: 15px; border-radius: 4px; margin: 15px 0; word-break: break-all; font-family: monospace; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>🔐 WealthWise Password Reset</h1>
                    </div>
                    <div class="content">
                        <p>Hello,</p>
                        <p>We received a request to reset your password for your WealthWise account. If you made this request, click the button below to reset your password:</p>
                        
                        <div style="text-align: center;">
                            <a href="{reset_url}" class="button">Reset My Password</a>
                        </div>
                        
                        <p>Or copy and paste this link into your browser:</p>
                        <div class="url-box">
                            {reset_url}
                        </div>
                        
                        <div class="warning">
                            <strong>⚠️ Important Security Information:</strong>
                            <ul>
                                <li><strong>This link will expire in 1 hour</strong></li>
                                <li>If you didn't request this reset, please ignore this email</li>
                                <li>For security, this link can only be used once</li>
                                <li>Never share this link with anyone</li>
                            </ul>
                        </div>
                        
                        <p>If you continue to have problems or didn't request this reset, please contact our support team.</p>
                        
                        <p>Best regards,<br>The WealthWise Team</p>
                    </div>
                    <div class="footer">
                        <p>This is an automated email from WealthWise. Please do not reply to this message.</p>
                        <p>© 2025 WealthWise. All rights reserved.</p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            text_body = f"""
WealthWise Password Reset Request

Hello,

We received a request to reset your password for your WealthWise account. 
If you made this request, click the link below to reset your password:

{reset_url}

Important Security Information:
- This link will expire in 1 hour
- If you didn't request this reset, please ignore this email
- For security, this link can only be used once
- Never share this link with anyone

If you continue to have problems or didn't request this reset, please contact our support team.

Best regards,
The WealthWise Team

This is an automated email from WealthWise. Please do not reply to this message.
© 2025 WealthWise. All rights reserved.
            """
            
            # Create message
            message = MIMEMultipart("alternative")
            message["Subject"] = subject
            message["From"] = self.from_email
            message["To"] = email
            
            # Add text and HTML parts
            text_part = MIMEText(text_body, "plain")
            html_part = MIMEText(html_body, "html")
            
            message.attach(text_part)
            message.attach(html_part)
            
            # Send email
            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.smtp_username, self.smtp_password)
                server.sendmail(self.from_email, email, message.as_string())
            
            logger.info(f"Password reset email sent successfully to {email}")
            return True
            
        except smtplib.SMTPAuthenticationError as e:
            logger.error(f"SMTP authentication failed: {e}")
            return False
        except smtplib.SMTPException as e:
            logger.error(f"SMTP error sending email to {email}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error sending reset email to {email}: {str(e)}")
            return False
    
    def _validate_password(self, password: str) -> bool:
        """
        Validate password strength
        
        Args:
            password: Password to validate
            
        Returns:
            bool: True if password meets requirements
        """
        if len(password) < 8:
            return False
        
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
        
        return all([has_upper, has_lower, has_digit, has_special])
    
    def cleanup_expired_tokens(self, db: Session) -> int:
        """
        Clean up expired password reset tokens
        
        Args:
            db: Database session
            
        Returns:
            int: Number of tokens cleaned up
        """
        try:
            expired_tokens = db.query(PasswordResetToken).filter(
                PasswordResetToken.expires_at < datetime.utcnow()
            ).all()
            
            count = len(expired_tokens)
            
            for token in expired_tokens:
                db.delete(token)
            
            db.commit()
            
            if count > 0:
                logger.info(f"Cleaned up {count} expired password reset tokens")
            
            return count
            
        except Exception as e:
            logger.error(f"Error cleaning up expired tokens: {e}")
            db.rollback()
            return 0