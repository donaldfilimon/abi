//! Local encryption/decryption for browser-uploaded sealed jobs; no account secrets on CLI.
#[allow(dead_code)]
mod support;
use abi_control_plane::{
    DeliveryPage, DeliveryPolicy, DurableSubmission, InferenceJob, JobBinding, JobView,
    NodeCertificate, RelayKey, ToolJob, VerifiedDeliveryStore, prepare_job, prepare_tool_job,
};
use std::{io::Read as _, path::Path};
use support::{Result, now, read_private, write_private};
fn main() {
    if run().is_err() {
        eprintln!(
            "Local job operation refused; check private files, signatures, expiry and result binding."
        );
        std::process::exit(1);
    }
}
fn run() -> Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    match args.as_slice() {
        [command,node,prompt,directory,model] if command=="prepare" || command=="prepare-durable" || command=="prepare-tool"=>{
            let node=Path::new(node);let directory=Path::new(directory);
            let cert:NodeCertificate=read_private(&node.join("node-certificate.json"))?;
            let authority=read_private(&node.join("authority-key.json"))?;
            let input_bytes=read_bounded(Path::new(prompt),8192)?;
            let seed=support::seed();let reply=RelayKey::from_seed(&seed)?;
            let submission=if command=="prepare-tool" {
                let input:ToolJob=serde_json::from_slice(&input_bytes)?;
                if input.revision!=*model {return Err("tool revision differs".into());}
                prepare_tool_job(&cert,&authority,&reply,&input,now()?)?
            } else {
                let input=InferenceJob {prompt:String::from_utf8(input_bytes)?,tokens:512,model:model.clone()};
                prepare_job(&cert,&authority,&reply,&input,now()?)?
            };
            support::directory(directory)?;write_private(&directory.join("reply-key.json"),&seed)?;
            write_private(&directory.join("job-binding.json"),&submission.binding)?;
            write_private(&directory.join("node-public-key.json"),&cert.claim.public_key)?;
            if command=="prepare" {write_private(&directory.join("submission.json"),&submission)?;} else {
                let delivery=DeliveryPolicy::for_submission(&submission,submission.binding.submitted_at+86400)?;
                write_private(&directory.join("delivery-policy.json"),&delivery)?;
                VerifiedDeliveryStore::create(&directory.join("delivery.sqlite"),&submission.binding,cert.claim.public_key,authority)?;
                write_private(&directory.join("submission.json"),&DurableSubmission{submission,delivery})?;
            }

            println!("Upload {} in the owning account's browser. Keep reply-key.json on this client.",directory.join("submission.json").display());
        }
        [command,directory,page] if command=="import"=>{
            let directory=Path::new(directory);let key=RelayKey::from_seed(&read_private(&directory.join("reply-key.json"))?)?;
            let binding:JobBinding=read_private(&directory.join("job-binding.json"))?;
            let policy:DeliveryPolicy=read_private(&directory.join("delivery-policy.json"))?;
            let page:DeliveryPage=serde_json::from_slice(&read_bounded(Path::new(page),384*1024)?)?;
            if page.receipt.binding!=binding || page.policy.policy!=policy{return Err("delivery scope or immutable policy differs".into());}
            let mut store=VerifiedDeliveryStore::open(&directory.join("delivery.sqlite"))?;
            let ack=store.import(&page,&key)?;
            write_ack(&directory.join("delivery-ack.json"),&serde_json::json!({"job_id":binding.job_id,"ack":ack}))?;
            println!("Encrypted packets verified and imported. Upload delivery-ack.json to acknowledge local receipt.");
        }
        [command,directory] if command=="open-durable"=>{
            let directory=Path::new(directory);let key=RelayKey::from_seed(&read_private(&directory.join("reply-key.json"))?)?;
            let store=VerifiedDeliveryStore::open(&directory.join("delivery.sqlite"))?;
            println!("{}",String::from_utf8(store.output(&key)?)?);
        }
        [command,directory,result] if command=="open" || command=="open-stream"=>{
            let directory=Path::new(directory);let key=RelayKey::from_seed(&read_private(&directory.join("reply-key.json"))?)?;
            let binding:JobBinding=read_private(&directory.join("job-binding.json"))?;
            let view:JobView=serde_json::from_slice(&read_bounded(Path::new(result),768*1024)?)?;
            if view.receipt.binding!=binding {return Err("result scope differs from submitted job".into());}
            if command=="open-stream" {
                let node_key=read_private(&directory.join("node-public-key.json"))?;
                let mut bytes=Vec::new();
                for (sequence,frame) in view.frames.iter().enumerate() {
                    if frame.sequence as usize!=sequence {return Err("stream sequence differs".into());}
                    bytes.extend(frame.open(&binding,&node_key,&key)?);
                    if bytes.len()>binding.max_result_bytes {return Err("stream exceeds bound".into());}
                }
                println!("Verified partial stream (remote state: {}):",view.receipt.state);
                println!("{}",String::from_utf8(bytes)?);return Ok(());
            }
            let bytes=key.open(&binding.aad("result")?,&view.result.ok_or("result unavailable")?)?;
            let result:abi_worker::JobResult=serde_json::from_slice(&bytes)?;
            if !result.verify_digest() || result.job_id.to_string()!=binding.job_id {return Err("result commitment differs".into());}
            println!("Local result status: {:?}",result.status);
            for chunk in result.chunks {print!("{}",String::from_utf8(chunk.bytes)?);}println!();
        }
        _=>return Err("usage: relay_job prepare|prepare-durable NODE_DIRECTORY PROMPT_FILE NEW_JOB_DIRECTORY MODEL_REVISION | prepare-tool NODE_DIRECTORY TOOL_JOB.json NEW_JOB_DIRECTORY TOOL_REVISION | import JOB_DIRECTORY ENCRYPTED_PAGE.json | open-durable JOB_DIRECTORY | open JOB_DIRECTORY ENCRYPTED_RESULT.json | open-stream JOB_DIRECTORY ENCRYPTED_OUTPUT.json".into()),
    }
    Ok(())
}
fn read_bounded(path: &Path, max: usize) -> Result<Vec<u8>> {
    let file = std::fs::File::open(path)?;
    if !file.metadata()?.is_file() {
        return Err("regular input file required".into());
    }
    let mut bytes = Vec::new();
    file.take(u64::try_from(max)? + 1).read_to_end(&mut bytes)?;
    if bytes.len() > max {
        return Err("input exceeds limit".into());
    }
    Ok(bytes)
}

fn write_ack(path: &Path, value: &impl serde::Serialize) -> Result<()> {
    match std::fs::symlink_metadata(path) {
        Ok(_) => {
            let _: serde_json::Value = read_private(path)?;
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(e.into()),
    }
    let temp = path.with_file_name(format!(".delivery-ack-{}.json", uuid::Uuid::new_v4()));
    write_private(&temp, value)?;
    std::fs::rename(&temp, path)?;
    std::fs::File::open(path.parent().ok_or("missing parent")?)?.sync_all()?;
    Ok(())
}
