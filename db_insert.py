from database import get_db
from models import Client, JobSeeker
from datetime import datetime

def save_client(data):
    db_gen = get_db()
    db = next(db_gen)
    try:
        new_client = Client(
            user_id="client",
            name=data.get("name"),
            email=data.get("email"),
            phone=data.get("phone"),
            tech_stack=data.get("tech_stack"),
            project_description=data.get("project_description"),
        )
        db.add(new_client)
        db.commit()
        db.refresh(new_client)
        print(f"✅ Client info saved (ID: {new_client.id})")
    except Exception as e:
        db.rollback()
        print("❌ Error saving client info:", repr(e))
    finally:
        db.close()


def save_jobseeker(data):
    db_gen = get_db()
    db = next(db_gen)
    try:
        new_job = JobSeeker(
            user_id="JobSeeker",
            name=data.get("name"),
            email=data.get("email"),
            resume_link=data.get("resume_link"),
            skills=data.get("skills")
        )
        db.add(new_job)
        db.commit()
        db.refresh(new_job)
        print(f"✅ Job Seeker info saved (ID: {new_job.id})")
    except Exception as e:
        db.rollback()
        print("❌ Error saving job seeker info:", repr(e))
    finally:
        db.close()
